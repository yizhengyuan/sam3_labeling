#!/usr/bin/env python3
"""
SAM 3 Video Tracking Script
使用 SAM3 进行视频目标追踪，输出 Label Studio 兼容的 JSON 格式

支持:
- CUDA (NVIDIA GPU) - 使用完整的视频追踪器
- MPS (Apple Metal) - 使用图像处理器逐帧处理
- CPU - 使用图像处理器逐帧处理（较慢）

使用前请确保：
1. 已安装 SAM3: pip install -e /path/to/sam3_repo
2. 已登录 Hugging Face: huggingface-cli login
3. 已申请 SAM3 模型访问权限: https://huggingface.co/facebook/sam3
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from collections import defaultdict

# SORT 追踪器依赖
try:
    from filterpy.kalman import KalmanFilter
    from scipy.optimize import linear_sum_assignment
    SORT_AVAILABLE = True
except ImportError:
    SORT_AVAILABLE = False
    print("⚠️ filterpy 或 scipy 未安装，将使用简单 IoU 匹配")
    print("   安装: pip install filterpy scipy")

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

# IoU 跟踪相关函数
def calculate_iou(box1, box2):
    """
    计算两个框的 IoU (Intersection over Union)
    
    Args:
        box1, box2: 格式为 (x1, y1, x2, y2) 的边界框
    
    Returns:
        IoU 值 (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def match_detections_to_tracks(detections, active_tracks, iou_threshold=0.3):
    """
    使用 IoU 匹配当前帧的检测结果与已有的轨迹
    
    Args:
        detections: 当前帧的检测结果 [(label, box, score), ...]
        active_tracks: 已有的轨迹 {track_id: {"label": str, "last_box": box, "last_frame": int}, ...}
        iou_threshold: IoU 匹配阈值
    
    Returns:
        matches: [(detection_idx, track_id), ...]
        unmatched_detections: [detection_idx, ...]
    """
    if not detections or not active_tracks:
        return [], list(range(len(detections)))
    
    matches = []
    used_tracks = set()
    unmatched_detections = []
    
    # 按照检测分数从高到低排序，优先匹配高置信度检测
    sorted_det_indices = sorted(range(len(detections)), 
                                 key=lambda i: detections[i][2], 
                                 reverse=True)
    
    for det_idx in sorted_det_indices:
        det_label, det_box, det_score = detections[det_idx]
        best_iou = 0
        best_track_id = None
        
        for track_id, track_info in active_tracks.items():
            if track_id in used_tracks:
                continue
            # 只匹配相同类别的目标
            if track_info["label"] != det_label:
                continue
            
            iou = calculate_iou(det_box, track_info["last_box"])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_track_id = track_id
        
        if best_track_id is not None:
            matches.append((det_idx, best_track_id))
            used_tracks.add(best_track_id)
        else:
            unmatched_detections.append(det_idx)
    
    return matches, unmatched_detections


# ==================== SORT 追踪器实现 ====================

class KalmanBoxTracker:
    """
    使用卡尔曼滤波追踪单个目标的边界框
    状态向量: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    """
    count = 0
    
    def __init__(self, bbox, label, score=1.0):
        """
        初始化追踪器
        
        Args:
            bbox: [x1, y1, x2, y2] 格式的边界框
            label: 目标类别
            score: 检测置信度
        """
        if not SORT_AVAILABLE:
            raise ImportError("filterpy not available")
        
        # 状态向量: [x, y, s, r, vx, vy, vs]
        # x, y: 中心坐标, s: 面积, r: 宽高比, v*: 速度
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
        
        # 测量噪声
        self.kf.R[2:, 2:] *= 10.
        
        # 协方差矩阵
        self.kf.P[4:, 4:] *= 1000.  # 速度的初始不确定性
        self.kf.P *= 10.
        
        # 过程噪声
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
        """
        将 [x1, y1, x2, y2] 转换为 [x_center, y_center, area, aspect_ratio]
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))
    
    @staticmethod
    def _z_to_bbox(z):
        """
        将 [x_center, y_center, area, aspect_ratio] 转换回 [x1, y1, x2, y2]
        """
        w = np.sqrt(z[2] * z[3])
        h = z[2] / w if w > 0 else 0
        return np.array([
            z[0] - w / 2.,
            z[1] - h / 2.,
            z[0] + w / 2.,
            z[1] + h / 2.
        ]).flatten()
    
    def update(self, bbox, score=None):
        """用新观测更新状态"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
        if score is not None:
            self.score = score
    
    def predict(self):
        """预测下一帧位置"""
        # 防止面积变为负数
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
        """获取当前状态的边界框"""
        return self._z_to_bbox(self.kf.x)


class SORTTracker:
    """
    SORT (Simple Online and Realtime Tracking) 追踪器
    
    特点：
    - 使用卡尔曼滤波预测目标位置
    - 使用匈牙利算法进行最优匹配
    - 支持多类别目标追踪
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        初始化追踪器
        
        Args:
            max_age: 目标丢失后保留的最大帧数
            min_hits: 连续命中多少次才算有效轨迹
            iou_threshold: IoU 匹配阈值
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
    
    def update(self, detections):
        """
        更新追踪器
        
        Args:
            detections: [(label, box, score), ...] 当前帧的检测结果
                        box 格式: (x1, y1, x2, y2)
        
        Returns:
            tracks: [(label, box, track_id, score), ...] 当前帧的追踪结果
        """
        self.frame_count += 1
        
        # 预测所有现有轨迹的新位置
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # 删除无效轨迹
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        # 匹配检测结果和轨迹
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, self.trackers, self.iou_threshold
        )
        
        # 更新匹配的轨迹
        for d, t in matched:
            label, box, score = detections[d]
            self.trackers[t].update(box, score)
        
        # 为未匹配的检测创建新轨迹
        for d in unmatched_dets:
            label, box, score = detections[d]
            trk = KalmanBoxTracker(box, label, score)
            self.trackers.append(trk)
        
        # 返回有效轨迹
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # 只返回近期有更新且达到最小命中次数的轨迹
            if (trk.time_since_update < 1) and \
               (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append((trk.label, tuple(d), trk.id, trk.score))
            i -= 1
            # 删除长时间未更新的轨迹
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        return ret
    
    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        """
        使用匈牙利算法将检测结果与轨迹关联
        
        Returns:
            matches: [[det_idx, trk_idx], ...]
            unmatched_detections: [det_idx, ...]
            unmatched_trackers: [trk_idx, ...]
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0, 5), dtype=int), np.arange(len(trackers))
        
        # 构建 IoU 矩阵
        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            det_label, det_box, det_score = det
            for t, trk in enumerate(trackers):
                # 只匹配相同类别
                if det_label == trk.label:
                    iou_matrix[d, t] = calculate_iou(det_box, trk.get_state())
        
        # 使用匈牙利算法求解最优匹配
        if min(iou_matrix.shape) > 0:
            # 转换为代价矩阵（1 - IoU）
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            matched_indices = np.array(list(zip(row_indices, col_indices)))
        else:
            matched_indices = np.empty((0, 2), dtype=int)
        
        # 找出未匹配的检测
        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0] if len(matched_indices) > 0 else True:
                unmatched_detections.append(d)
        
        # 找出未匹配的轨迹
        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1] if len(matched_indices) > 0 else True:
                unmatched_trackers.append(t)
        
        # 过滤掉低 IoU 的匹配
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# 检查 SAM3 是否可用
SAM3_AVAILABLE = False
SAM3_VIDEO_AVAILABLE = False

try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
    
    # 检查是否支持视频追踪（需要 CUDA）
    if torch.cuda.is_available():
        SAM3_VIDEO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ SAM3 未安装或导入失败: {e}")
    print("请运行: pip install -e /path/to/sam3_repo")


def run_video_tracking_cuda(
    video_path: str,
    text_prompt: str,
    output_path: str,
    sample_rate: int = 1
):
    """
    使用 CUDA 的完整视频追踪（需要 NVIDIA GPU）
    """
    from sam3.visualization_utils import prepare_masks_for_visualization
    
    print(f"🎬 加载视频: {video_path}")
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"   帧率: {fps}, 总帧数: {frame_count}")
    
    gpus_to_use = list(range(torch.cuda.device_count()))
    print(f"🖥️ 使用 GPU: {gpus_to_use}")
    
    # 构建 SAM3 视频预测器
    print("🔧 加载 SAM3 模型...")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    
    # 开始会话
    print("📹 开始视频会话...")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    
    # 添加文本提示
    print(f"🏷️ 添加文本提示: '{text_prompt}'")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )
    
    # 传播追踪
    print("🔄 传播追踪到所有帧...")
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    
    print(f"   追踪完成，共 {len(outputs_per_frame)} 帧")
    
    # 关闭会话
    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    predictor.shutdown()
    
    # 转换为 Label Studio 格式
    return convert_outputs_to_label_studio(
        outputs_per_frame, video_path, fps, text_prompt, output_path
    )


def draw_box_on_frame(frame, x1, y1, x2, y2, label, obj_id, color):
    """在帧上绘制边界框"""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 绘制标签背景
    text = f"{label} #{obj_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(frame, (x1, y1 - text_height - 6), (x1 + text_width + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 3), font, font_scale, (255, 255, 255), thickness)
    
    return frame


def run_video_tracking_mps_cpu(
    video_path: str,
    text_prompts: List[str],  # 改为支持多个提示词
    output_path: str,
    device: str = "mps",
    sample_rate: int = 5,
    checkpoint_path: str = None,
    generate_video: bool = True,  # 是否生成标注视频
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.15,  # IoU 匹配阈值
    debug: bool = False,  # 是否显示调试信息
    use_sort: bool = True,  # 是否使用 SORT 追踪器
    max_age: int = None,  # SORT: 目标丢失后保留的最大帧数
    min_hits: int = 3  # SORT: 连续命中多少次才算有效轨迹
):
    """
    使用 MPS 或 CPU 的逐帧处理（适用于 Mac 或无 GPU 环境）
    支持多个文本提示，每个目标独立标注
    
    追踪模式:
    - SORT 追踪器 (use_sort=True): 使用卡尔曼滤波 + 匈牙利算法，ID 更稳定
    - 简单 IoU 匹配 (use_sort=False): 原始方式，适用于 filterpy 未安装的情况
    """
    print(f"🎬 加载视频: {video_path}")
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   帧率: {fps}, 总帧数: {frame_count}, 分辨率: {width}x{height}")
    print(f"🖥️ 使用设备: {device}")
    print(f"📊 采样率: 每 {sample_rate} 帧处理一次")
    print(f"🏷️ 检测目标: {text_prompts}")
    
    # 决定使用哪种追踪模式
    use_sort_tracker = use_sort and SORT_AVAILABLE
    if use_sort and not SORT_AVAILABLE:
        print("⚠️ SORT 追踪器不可用（filterpy 未安装），使用简单 IoU 匹配")
    
    if use_sort_tracker:
        # 初始化 SORT 追踪器
        if max_age is None:
            max_age = int(fps * 2)  # 默认丢失 2 秒后删除轨迹
        sort_tracker = SORTTracker(
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold
        )
        print(f"🔄 使用 SORT 追踪器 (卡尔曼滤波 + 匈牙利算法)")
        print(f"   参数: max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
    else:
        print(f"🔄 使用简单 IoU 匹配追踪")
    
    # 构建 SAM3 图像模型
    print("🔧 加载 SAM3 模型...")
    if checkpoint_path:
        print(f"   使用本地模型: {checkpoint_path}")
        model = build_sam3_image_model(device=device, checkpoint_path=checkpoint_path, load_from_HF=False)
    else:
        model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    
    # 准备输出视频
    video_output_path = None
    video_writer = None
    temp_video_path = None
    if generate_video:
        video_output_path = output_path.replace('.json', '_annotated.mp4')
        # 使用临时文件，之后用 ffmpeg 重新编码以获得更好的兼容性
        temp_video_path = output_path.replace('.json', '_temp.avi')
        # 使用 XVID 编码，兼容性更好
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    # 存储所有检测结果
    # 结构: {track_id: {"label": str, "global_id": int, "frames": {frame_idx: box_data}}}
    all_results = {}
    
    # 简单 IoU 匹配模式的变量
    active_tracks = {}  # {track_id: {"label": str, "last_box": (x1,y1,x2,y2), "last_frame": int}}
    next_track_id = 0
    max_missing_frames = int(fps * 2)  # 最多允许丢失 2 秒
    
    # 处理视频帧
    print(f"🔄 处理视频帧...")
    
    frame_idx = 0
    processed_count = 0
    
    # 当前帧的标注（用于视频绘制）
    current_frame_annotations = []
    
    # 重新打开视频
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_annotations = []  # 当前帧的所有标注
        
        # 采样帧进行检测
        if frame_idx % sample_rate == 0:
            # 转换为 PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 收集当前帧所有检测结果
            current_detections = []  # [(label, box, score), ...]
            
            # 对每个文本提示进行检测
            for text_prompt in text_prompts:
                # 设置图像
                state = processor.set_image(pil_image)
                
                # 使用文本提示进行检测
                output = processor.set_text_prompt(state=state, prompt=text_prompt)
                
                boxes = output.get("boxes", [])
                scores = output.get("scores", [])
                
                # 收集检测结果
                for i in range(len(boxes)):
                    box = boxes[i]
                    score = scores[i] if i < len(scores) else 0.5
                    
                    if score < confidence_threshold:
                        continue
                    
                    # box 格式: [x1, y1, x2, y2] 像素坐标
                    x1, y1, x2, y2 = box.cpu().numpy()
                    current_detections.append((text_prompt, (float(x1), float(y1), float(x2), float(y2)), float(score)))
            
            # ========== 追踪处理 ==========
            if use_sort_tracker:
                # 使用 SORT 追踪器
                tracks = sort_tracker.update(current_detections)
                
                if debug and tracks:
                    print(f"   [帧 {frame_idx}] 检测到 {len(current_detections)} 个目标, "
                          f"追踪到 {len(tracks)} 个")
                
                for label, box, track_id, score in tracks:
                    x1, y1, x2, y2 = box
                    
                    # 确保结果记录存在
                    if track_id not in all_results:
                        all_results[track_id] = {
                            "label": label,
                            "global_id": track_id,
                            "frames": {}
                        }
                    
                    # 转换为 Label Studio 格式 (百分比 0-100)
                    box_data = {
                        "x": x1 / width * 100,
                        "y": y1 / height * 100,
                        "width": (x2 - x1) / width * 100,
                        "height": (y2 - y1) / height * 100,
                        "score": score,
                        "time": frame_idx / fps,
                        "pixel_box": (int(x1), int(y1), int(x2), int(y2))
                    }
                    all_results[track_id]["frames"][frame_idx] = box_data
                    
                    # 添加到当前帧的标注列表
                    frame_annotations.append({
                        "label": label,
                        "obj_id": track_id,
                        "pixel_box": (int(x1), int(y1), int(x2), int(y2)),
                        "color": COLORS[track_id % len(COLORS)]
                    })
            else:
                # 使用简单 IoU 匹配
                matches, unmatched = match_detections_to_tracks(
                    current_detections, active_tracks, iou_threshold
                )
                
                if debug and (matches or unmatched):
                    print(f"   [帧 {frame_idx}] 检测到 {len(current_detections)} 个目标, "
                          f"匹配 {len(matches)} 个, 新增 {len(unmatched)} 个")
                
                # 更新匹配的轨迹
                for det_idx, track_id in matches:
                    label, box, score = current_detections[det_idx]
                    x1, y1, x2, y2 = box
                    
                    active_tracks[track_id]["last_box"] = box
                    active_tracks[track_id]["last_frame"] = frame_idx
                    
                    box_data = {
                        "x": x1 / width * 100,
                        "y": y1 / height * 100,
                        "width": (x2 - x1) / width * 100,
                        "height": (y2 - y1) / height * 100,
                        "score": score,
                        "time": frame_idx / fps,
                        "pixel_box": (int(x1), int(y1), int(x2), int(y2))
                    }
                    all_results[track_id]["frames"][frame_idx] = box_data
                    
                    frame_annotations.append({
                        "label": label,
                        "obj_id": all_results[track_id]["global_id"],
                        "pixel_box": (int(x1), int(y1), int(x2), int(y2)),
                        "color": COLORS[all_results[track_id]["global_id"] % len(COLORS)]
                    })
                
                # 为未匹配的检测创建新轨迹
                for det_idx in unmatched:
                    label, box, score = current_detections[det_idx]
                    x1, y1, x2, y2 = box
                    
                    track_id = next_track_id
                    next_track_id += 1
                    
                    active_tracks[track_id] = {
                        "label": label,
                        "last_box": box,
                        "last_frame": frame_idx
                    }
                    
                    all_results[track_id] = {
                        "label": label,
                        "global_id": track_id,
                        "frames": {}
                    }
                    
                    box_data = {
                        "x": x1 / width * 100,
                        "y": y1 / height * 100,
                        "width": (x2 - x1) / width * 100,
                        "height": (y2 - y1) / height * 100,
                        "score": score,
                        "time": frame_idx / fps,
                        "pixel_box": (int(x1), int(y1), int(x2), int(y2))
                    }
                    all_results[track_id]["frames"][frame_idx] = box_data
                    
                    frame_annotations.append({
                        "label": label,
                        "obj_id": track_id,
                        "pixel_box": (int(x1), int(y1), int(x2), int(y2)),
                        "color": COLORS[track_id % len(COLORS)]
                    })
                
                # 清理长时间未更新的轨迹
                tracks_to_remove = [
                    tid for tid, info in active_tracks.items()
                    if frame_idx - info["last_frame"] > max_missing_frames
                ]
                for tid in tracks_to_remove:
                    del active_tracks[tid]
            
            # 保存当前帧标注
            current_frame_annotations = frame_annotations.copy()
            
            processed_count += 1
            if processed_count % 10 == 0:
                if use_sort_tracker:
                    print(f"   已处理 {processed_count} 帧，活跃轨迹: {len(sort_tracker.trackers)}")
                else:
                    print(f"   已处理 {processed_count} 帧，活跃轨迹: {len(active_tracks)}")
        else:
            # 非采样帧：使用上一次的标注（或者使用 SORT 预测）
            if use_sort_tracker and SORT_AVAILABLE:
                # SORT 模式：可以使用卡尔曼滤波预测位置
                for trk in sort_tracker.trackers:
                    if trk.time_since_update == 0:  # 只显示最近更新过的轨迹
                        pred_box = trk.get_state()
                        x1, y1, x2, y2 = pred_box
                        frame_annotations.append({
                            "label": trk.label,
                            "obj_id": trk.id,
                            "pixel_box": (int(x1), int(y1), int(x2), int(y2)),
                            "color": COLORS[trk.id % len(COLORS)]
                        })
            else:
                # 简单模式：使用上一帧的标注
                frame_annotations = current_frame_annotations.copy()
        
        # 生成标注视频帧
        if video_writer is not None:
            # 在帧上绘制所有标注
            for ann in frame_annotations:
                x1, y1, x2, y2 = ann["pixel_box"]
                frame = draw_box_on_frame(
                    frame, x1, y1, x2, y2,
                    ann["label"], ann["obj_id"], ann["color"]
                )
            
            # 显示帧号和追踪模式
            tracker_mode = "SORT" if use_sort_tracker else "IoU"
            cv2.putText(frame, f"Frame: {frame_idx} [{tracker_mode}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            video_writer.write(frame)
        
        frame_idx += 1
    
    cap.release()
    if video_writer is not None:
        video_writer.release()
        
        # 使用 ffmpeg 重新编码为 H.264 MP4，获得更好的兼容性
        if temp_video_path and os.path.exists(temp_video_path):
            print("🔄 正在优化视频编码...")
            import subprocess
            try:
                # 使用 ffmpeg 转换为 H.264 编码
                cmd = [
                    'ffmpeg', '-y', '-i', temp_video_path,
                    '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                    '-pix_fmt', 'yuv420p',  # 确保兼容性
                    video_output_path
                ]
                subprocess.run(cmd, capture_output=True, check=True)
                os.remove(temp_video_path)  # 删除临时文件
                print("   视频编码优化完成")
            except (subprocess.CalledProcessError, FileNotFoundError):
                # 如果 ffmpeg 不可用，直接重命名临时文件
                print("   ffmpeg 不可用，使用原始编码")
                import shutil
                shutil.move(temp_video_path, video_output_path)
    
    print(f"   处理完成，共处理 {processed_count} 帧")
    
    # 转换为 Label Studio 格式
    ls_results = []
    for obj_key, obj_data in all_results.items():
        frames_data = obj_data["frames"]
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
                    "labels": [obj_data["label"]]
                },
                "id": f"obj_{obj_data['global_id']}"
            })
    
    # 保存 JSON 结果
    tracker_version = "SAM3-SORT" if use_sort_tracker else "SAM3-IoU"
    output_data = [{
        "data": {
            "video": f"/data/local-files/?d={os.path.basename(video_path)}"
        },
        "predictions": [{
            "result": ls_results,
            "model_version": tracker_version
        }]
    }]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ JSON 结果已保存到: {output_path}")
    print(f"   共追踪到 {len(ls_results)} 个目标")
    
    if video_output_path:
        print(f"✅ 标注视频已保存到: {video_output_path}")


def convert_outputs_to_label_studio(
    outputs_per_frame: Dict,
    video_path: str,
    fps: float,
    text_prompt: str,
    output_path: str
):
    """将 SAM3 输出转换为 Label Studio 格式"""
    from sam3.visualization_utils import prepare_masks_for_visualization
    
    print("📝 转换为 Label Studio 格式...")
    formatted_outputs = prepare_masks_for_visualization(outputs_per_frame)
    
    all_results = {}  # {obj_id: [sequence]}
    
    for frame_idx, frame_output in formatted_outputs.items():
        if frame_output is None:
            continue
        
        boxes = frame_output.get("boxes", [])
        scores = frame_output.get("scores", [])
        obj_ids = frame_output.get("obj_ids", [])
        
        for i, obj_id in enumerate(obj_ids):
            if i < len(boxes) and boxes[i] is not None:
                box = boxes[i]
                x = float(box[0]) * 100
                y = float(box[1]) * 100
                width = float(box[2] - box[0]) * 100
                height = float(box[3] - box[1]) * 100
                score = float(scores[i]) if i < len(scores) else 1.0
                
                str_obj_id = f"obj_{obj_id}"
                if str_obj_id not in all_results:
                    all_results[str_obj_id] = []
                
                all_results[str_obj_id].append({
                    "frame": frame_idx,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "rotation": 0,
                    "time": frame_idx / fps,
                    "enabled": True
                })
    
    ls_results = []
    for obj_id, sequence in all_results.items():
        if sequence:
            ls_results.append({
                "from_name": "box",
                "to_name": "video",
                "type": "videorectangle",
                "value": {
                    "sequence": sorted(sequence, key=lambda x: x["frame"]),
                    "labels": [text_prompt]
                },
                "id": obj_id
            })
    
    output_data = [{
        "data": {
            "video": f"/data/local-files/?d={os.path.basename(video_path)}"
        },
        "predictions": [{
            "result": ls_results,
            "model_version": "SAM3"
        }]
    }]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存到: {output_path}")
    print(f"   共追踪到 {len(ls_results)} 个目标")


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 视频追踪 - 使用文本提示追踪视频中的目标"
    )
    parser.add_argument(
        "video_path",
        help="输入视频路径 (MP4 或 JPEG 帧目录)"
    )
    parser.add_argument(
        "--text", "-t",
        required=True,
        nargs="+",  # 支持多个文本提示
        help="文本提示，描述要追踪的目标，可以指定多个 (如 -t car 'traffic sign')"
    )
    parser.add_argument(
        "--output", "-o",
        default="SAM3_output/tracking_result.json",
        help="输出 JSON 路径 (默认: SAM3_output/tracking_result.json)"
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="计算设备 (默认: 自动选择)"
    )
    parser.add_argument(
        "--sample-rate", "-s",
        type=int,
        default=5,
        help="采样率，每 N 帧处理一次 (仅用于 MPS/CPU 模式，默认: 5)"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        default=None,
        help="本地模型 checkpoint 路径 (如 checkpoints/sam3/sam3.pt)"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不生成标注视频，只输出 JSON"
    )
    parser.add_argument(
        "--confidence", 
        type=float,
        default=0.3,
        help="置信度阈值 (默认: 0.3)"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.15,
        help="IoU 匹配阈值，越低越容易匹配同一目标 (默认: 0.15)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示详细的跟踪调试信息"
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="禁用 SORT 追踪器，使用简单 IoU 匹配（不推荐）"
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=None,
        help="SORT: 目标丢失后保留的最大帧数（默认: fps * 2）"
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="SORT: 连续命中多少次才算有效轨迹（默认: 3）"
    )
    
    args = parser.parse_args()
    
    if not SAM3_AVAILABLE:
        print("❌ SAM3 不可用，无法运行")
        sys.exit(1)
    
    # 检查输入文件
    if not os.path.exists(args.video_path):
        print(f"❌ 视频文件不存在: {args.video_path}")
        sys.exit(1)
    
    # 自动选择设备
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    # 根据设备选择处理方式
    if args.device == "cuda" and SAM3_VIDEO_AVAILABLE:
        run_video_tracking_cuda(
            video_path=args.video_path,
            text_prompt=args.text[0],  # CUDA 模式暂时只支持单个提示
            output_path=args.output,
            sample_rate=args.sample_rate
        )
    else:
        print(f"⚠️ 使用逐帧处理模式 (设备: {args.device})")
        run_video_tracking_mps_cpu(
            video_path=args.video_path,
            text_prompts=args.text,  # 支持多个提示
            output_path=args.output,
            device=args.device,
            sample_rate=args.sample_rate,
            checkpoint_path=args.checkpoint,
            generate_video=not args.no_video,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou_threshold,
            debug=args.debug,
            use_sort=not args.no_sort,
            max_age=args.max_age,
            min_hits=args.min_hits
        )


if __name__ == "__main__":
    main()
