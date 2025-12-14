#!/usr/bin/env python3
"""
使用 SORT 追踪器重新处理已有的检测结果
将散乱的检测框重新关联，合并为稳定的轨迹

用法:
    python3 scripts/retrack_with_sort.py \
        SAM3_output/clip_000_every_frame.json \
        --output SAM3_output/clip_000_sort_retracked.json \
        --video data/D1_video_clips/D1_rand11-15_clip_000.mp4
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

# SORT 追踪器依赖
try:
    from filterpy.kalman import KalmanFilter
    from scipy.optimize import linear_sum_assignment
    SORT_AVAILABLE = True
except ImportError:
    print("❌ 请先安装依赖: pip install filterpy scipy")
    sys.exit(1)

# 颜色列表（BGR格式）用于可视化
COLORS = [
    (0, 255, 0),    # 绿色
    (255, 0, 0),    # 蓝色
    (0, 0, 255),    # 红色
    (255, 255, 0),  # 青色
    (255, 0, 255),  # 紫色
    (0, 255, 255),  # 黄色
    (128, 255, 0),  # 浅绿
    (255, 128, 0),  # 浅蓝
    (128, 0, 255),  # 粉色
    (0, 128, 255),  # 橙色
]


def calculate_iou(box1, box2):
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


class KalmanBoxTracker:
    """使用卡尔曼滤波追踪单个目标的边界框"""
    count = 0
    
    def __init__(self, bbox, label, score=1.0):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # 状态转移矩阵
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # 噪声参数
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # 初始化状态
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.label = label
        self.score = score
    
    @staticmethod
    def _bbox_to_z(bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _z_to_bbox(z):
        w = np.sqrt(z[2] * z[3])
        h = z[2] / w if w > 0 else 0
        return np.array([
            z[0] - w / 2.,
            z[1] - h / 2.,
            z[0] + w / 2.,
            z[1] + h / 2.
        ]).flatten()
    
    def update(self, bbox, score=None):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
        if score is not None:
            self.score = score
    
    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.kf.x))
        return self.history[-1]
    
    def get_state(self):
        return self._z_to_bbox(self.kf.x)


class SORTTracker:
    """SORT 追踪器"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, detections):
        """
        Args:
            detections: [(label, box, score), ...]
        Returns:
            tracks: [(label, box, track_id, score), ...]
        """
        self.frame_count += 1
        
        # 预测所有现有轨迹
        for trk in self.trackers:
            trk.predict()
        
        # 删除无效轨迹
        self.trackers = [t for t in self.trackers if not np.any(np.isnan(t.get_state()))]
        
        # 匹配检测和轨迹
        matched, unmatched_dets, _ = self._associate(detections)
        
        # 更新匹配的轨迹
        for d, t in matched:
            label, box, score = detections[d]
            self.trackers[t].update(box, score)
        
        # 为未匹配的检测创建新轨迹
        for d in unmatched_dets:
            label, box, score = detections[d]
            self.trackers.append(KalmanBoxTracker(box, label, score))
        
        # 返回有效轨迹
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append((trk.label, tuple(d), trk.id, trk.score))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        return ret
    
    def _associate(self, detections):
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.trackers)))
        
        # 构建 IoU 矩阵
        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            det_label, det_box, _ = det
            for t, trk in enumerate(self.trackers):
                if det_label == trk.label:
                    iou_matrix[d, t] = calculate_iou(det_box, trk.get_state())
        
        # 匈牙利算法
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = list(zip(row_indices, col_indices))
        
        unmatched_dets = [d for d in range(len(detections)) 
                         if d not in [m[0] for m in matched_indices]]
        unmatched_trks = [t for t in range(len(self.trackers)) 
                         if t not in [m[1] for m in matched_indices]]
        
        # 过滤低 IoU 匹配
        matches = []
        for d, t in matched_indices:
            if iou_matrix[d, t] < self.iou_threshold:
                unmatched_dets.append(d)
                unmatched_trks.append(t)
            else:
                matches.append((d, t))
        
        return matches, unmatched_dets, unmatched_trks


def load_detections_from_json(json_path: str) -> Dict[int, List]:
    """
    从 Label Studio JSON 格式加载检测结果
    
    Returns:
        {frame_idx: [(label, box, score), ...], ...}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detections_by_frame = defaultdict(list)
    
    # 遍历所有标注结果
    for item in data:
        predictions = item.get("predictions", [])
        for pred in predictions:
            results = pred.get("result", [])
            for result in results:
                if result.get("type") != "videorectangle":
                    continue
                
                value = result.get("value", {})
                labels = value.get("labels", [])
                label = labels[0] if labels else "unknown"
                sequence = value.get("sequence", [])
                
                for frame_data in sequence:
                    frame_idx = frame_data.get("frame", 0)
                    # 百分比坐标 (0-100)
                    x = frame_data.get("x", 0)
                    y = frame_data.get("y", 0)
                    w = frame_data.get("width", 0)
                    h = frame_data.get("height", 0)
                    
                    # 存储为百分比格式的 box (x1, y1, x2, y2)
                    box = (x, y, x + w, y + h)
                    score = frame_data.get("score", 0.5) if "score" in frame_data else 0.5
                    
                    detections_by_frame[frame_idx].append((label, box, score))
    
    return dict(detections_by_frame)


def draw_box_on_frame(frame, x1, y1, x2, y2, label, obj_id, color):
    """在帧上绘制边界框"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    text = f"{label} #{obj_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 3), font, font_scale, (255, 255, 255), thickness)
    
    return frame


def retrack_with_sort(
    input_json: str,
    output_json: str,
    video_path: str = None,
    max_age: int = 30,
    min_hits: int = 3,
    iou_threshold: float = 0.3,
    generate_video: bool = True,
    debug: bool = False
):
    """
    使用 SORT 重新追踪已有的检测结果
    """
    print(f"📂 加载检测结果: {input_json}")
    detections_by_frame = load_detections_from_json(input_json)
    
    if not detections_by_frame:
        print("❌ 没有找到检测结果")
        return
    
    frame_indices = sorted(detections_by_frame.keys())
    max_frame = max(frame_indices)
    total_detections = sum(len(d) for d in detections_by_frame.values())
    
    print(f"   帧范围: {min(frame_indices)} - {max_frame}")
    print(f"   总检测数: {total_detections}")
    
    # 获取视频信息
    width, height, fps = 1920, 1080, 25.0
    video_writer = None
    cap = None
    
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"📹 视频信息: {width}x{height} @ {fps:.1f}fps")
    
    # 初始化 SORT 追踪器
    # 重置 KalmanBoxTracker 计数器
    KalmanBoxTracker.count = 0
    
    tracker = SORTTracker(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold
    )
    
    print(f"🔄 使用 SORT 追踪器")
    print(f"   参数: max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
    
    # 准备输出视频
    temp_video_path = None
    video_output_path = None
    if generate_video and cap is not None:
        video_output_path = output_json.replace('.json', '_annotated.mp4')
        temp_video_path = output_json.replace('.json', '_temp.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # 存储追踪结果
    all_tracks = {}  # {track_id: {"label": str, "frames": {frame_idx: box_data}}}
    
    print(f"🔄 处理帧...")
    
    for frame_idx in range(max_frame + 1):
        # 获取当前帧的检测
        detections = detections_by_frame.get(frame_idx, [])
        
        # 将百分比坐标转换为像素坐标（用于追踪）
        pixel_detections = []
        for label, box, score in detections:
            x1, y1, x2, y2 = box
            px1 = x1 / 100 * width
            py1 = y1 / 100 * height
            px2 = x2 / 100 * width
            py2 = y2 / 100 * height
            pixel_detections.append((label, (px1, py1, px2, py2), score))
        
        # 更新追踪器
        tracks = tracker.update(pixel_detections)
        
        if debug and tracks:
            print(f"   [帧 {frame_idx}] 检测: {len(detections)}, 追踪: {len(tracks)}")
        
        # 保存追踪结果
        frame_annotations = []
        for label, box, track_id, score in tracks:
            x1, y1, x2, y2 = box
            
            if track_id not in all_tracks:
                all_tracks[track_id] = {
                    "label": label,
                    "frames": {}
                }
            
            # 转换回百分比
            box_data = {
                "x": x1 / width * 100,
                "y": y1 / height * 100,
                "width": (x2 - x1) / width * 100,
                "height": (y2 - y1) / height * 100,
                "time": frame_idx / fps
            }
            all_tracks[track_id]["frames"][frame_idx] = box_data
            
            frame_annotations.append({
                "label": label,
                "obj_id": track_id,
                "pixel_box": (int(x1), int(y1), int(x2), int(y2)),
                "color": COLORS[track_id % len(COLORS)]
            })
        
        # 生成视频帧
        if video_writer is not None and cap is not None:
            ret, frame = cap.read()
            if ret:
                for ann in frame_annotations:
                    x1, y1, x2, y2 = ann["pixel_box"]
                    frame = draw_box_on_frame(
                        frame, x1, y1, x2, y2,
                        ann["label"], ann["obj_id"], ann["color"]
                    )
                
                cv2.putText(frame, f"Frame: {frame_idx} [SORT]", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                video_writer.write(frame)
        
        if frame_idx % 50 == 0:
            print(f"   已处理 {frame_idx}/{max_frame} 帧")
    
    if cap is not None:
        cap.release()
    if video_writer is not None:
        video_writer.release()
        
        # 使用 ffmpeg 优化编码
        if temp_video_path and os.path.exists(temp_video_path):
            print("🔄 正在优化视频编码...")
            import subprocess
            try:
                cmd = [
                    'ffmpeg', '-y', '-i', temp_video_path,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                    video_output_path
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                os.remove(temp_video_path)
                print("   视频编码优化完成")
            except (subprocess.CalledProcessError, FileNotFoundError):
                import shutil
                shutil.move(temp_video_path, video_output_path)
    
    # 转换为 Label Studio 格式
    ls_results = []
    for track_id, track_data in all_tracks.items():
        frames_data = track_data["frames"]
        if not frames_data:
            continue
        
        sequence = []
        for fidx, data in sorted(frames_data.items()):
            sequence.append({
                "frame": fidx,
                "x": data["x"],
                "y": data["y"],
                "width": data["width"],
                "height": data["height"],
                "rotation": 0,
                "time": data["time"],
                "enabled": True
            })
        
        if sequence:
            ls_results.append({
                "from_name": "box",
                "to_name": "video",
                "type": "videorectangle",
                "value": {
                    "sequence": sequence,
                    "labels": [track_data["label"]]
                },
                "id": f"track_{track_id}"
            })
    
    # 保存 JSON 结果
    output_data = [{
        "data": {
            "video": f"/data/local-files/?d={os.path.basename(video_path) if video_path else 'video.mp4'}"
        },
        "predictions": [{
            "result": ls_results,
            "model_version": "SORT-Retracked"
        }]
    }]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 统计
    print(f"\n✅ 追踪完成!")
    print(f"   原始检测: {total_detections} 个框")
    print(f"   合并轨迹: {len(ls_results)} 条")
    print(f"   JSON 保存到: {output_json}")
    
    if video_output_path:
        print(f"   视频保存到: {video_output_path}")
    
    # 显示每个轨迹的信息
    print(f"\n📊 轨迹详情:")
    label_counts = defaultdict(int)
    for track_id, track_data in all_tracks.items():
        label = track_data["label"]
        frame_count = len(track_data["frames"])
        label_counts[label] += 1
        if debug:
            frames = sorted(track_data["frames"].keys())
            print(f"   Track #{track_id} [{label}]: {frame_count} 帧 (帧 {frames[0]}-{frames[-1]})")
    
    for label, count in sorted(label_counts.items()):
        print(f"   - {label}: {count} 个目标")


def main():
    parser = argparse.ArgumentParser(
        description="使用 SORT 追踪器重新处理检测结果"
    )
    parser.add_argument(
        "input_json",
        help="输入 JSON 文件路径（Label Studio 格式）"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出 JSON 路径（默认: 输入文件名_sort.json）"
    )
    parser.add_argument(
        "--video", "-v",
        default=None,
        help="原始视频路径（用于生成标注视频）"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="目标丢失后保留的最大帧数（默认: 30）"
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="连续命中多少次才算有效轨迹（默认: 3）"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU 匹配阈值（默认: 0.3）"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不生成标注视频"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示详细调试信息"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"❌ 输入文件不存在: {args.input_json}")
        sys.exit(1)
    
    if args.output is None:
        args.output = args.input_json.replace('.json', '_sort.json')
    
    retrack_with_sort(
        input_json=args.input_json,
        output_json=args.output,
        video_path=args.video,
        max_age=args.max_age,
        min_hits=args.min_hits,
        iou_threshold=args.iou_threshold,
        generate_video=not args.no_video,
        debug=args.debug
    )


if __name__ == "__main__":
    main()





