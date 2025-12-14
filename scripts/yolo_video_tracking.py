#!/usr/bin/env python3
"""
YOLO + ByteTrack 视频目标追踪脚本
快速检测并追踪视频中的目标，支持跨帧 ID 保持

支持的目标类别（COCO）：
- person (行人)
- car (汽车)
- motorcycle (摩托车)
- bicycle (自行车)
- truck (卡车)
- bus (公交车)
- traffic light (交通灯)
- stop sign (停止标志)
等 80 个类别

注意：COCO 模型不支持一般交通标志，需要使用专门的模型或 SAM3
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

# COCO 类别映射（部分）
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
    9: 'traffic light',
    11: 'stop sign',
}

# 你需要的类别
TARGET_CLASSES = {
    'person': '行人',
    'car': '汽车',
    'motorcycle': '摩托车',
    'bicycle': '自行车',
    'bus': '公交车',
    'truck': '卡车',
    'traffic light': '交通灯',
    'stop sign': '停止标志',
}

# 颜色映射 - 按类别固定颜色，不随 ID 变化
COLORS = {
    'person': (0, 255, 0),       # 绿色
    'car': (255, 0, 0),          # 蓝色
    'motorcycle': (0, 0, 255),   # 红色
    'bicycle': (255, 255, 0),    # 青色
    'bus': (255, 0, 255),        # 紫色
    'truck': (0, 255, 255),      # 黄色
    'traffic light': (128, 255, 0),
    'stop sign': (0, 128, 255),
}


def is_first_person_vehicle(box, frame_height, frame_width, cls_name):
    """
    判断是否为第一人称视角的车辆（骑行者自己的摩托车/自行车）
    
    特征：
    1. 位于画面底部（y > 60% 的高度）
    2. 面积较大（占画面 > 15%）
    3. 是摩托车或自行车
    """
    if cls_name not in ['motorcycle', 'bicycle']:
        return False
    
    x1, y1, x2, y2 = box
    box_height = y2 - y1
    box_width = x2 - x1
    box_area = box_height * box_width
    frame_area = frame_height * frame_width
    
    # 条件1：底部区域（中心点在下半部分）
    center_y = (y1 + y2) / 2
    is_bottom = center_y > frame_height * 0.5
    
    # 条件2：面积较大
    area_ratio = box_area / frame_area
    is_large = area_ratio > 0.1
    
    # 条件3：宽度较大（横跨画面）
    width_ratio = box_width / frame_width
    is_wide = width_ratio > 0.3
    
    return is_bottom and (is_large or is_wide)


def run_yolo_tracking(
    video_path: str,
    output_json: str,
    target_classes: List[str] = None,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.3,
    generate_video: bool = True,
    device: str = "mps"  # Mac 使用 MPS
):
    """
    使用 YOLO + ByteTrack 进行视频目标追踪
    
    Args:
        video_path: 输入视频路径
        output_json: 输出 JSON 路径
        target_classes: 要检测的类别列表
        model_name: YOLO 模型名称
        confidence: 置信度阈值
        generate_video: 是否生成标注视频
        device: 计算设备
    """
    from ultralytics import YOLO
    
    print(f"🎬 加载视频: {video_path}")
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"   帧率: {fps:.1f}, 总帧数: {frame_count}, 分辨率: {width}x{height}")
    print(f"   视频时长: {frame_count/fps:.1f} 秒")
    
    # 加载 YOLO 模型
    print(f"🔧 加载 YOLO 模型: {model_name}")
    model = YOLO(model_name)
    
    # 设置要检测的类别
    if target_classes:
        # 获取类别 ID
        class_names = model.names
        class_ids = [k for k, v in class_names.items() if v in target_classes]
        print(f"🏷️ 检测类别: {target_classes}")
    else:
        class_ids = None
        print(f"🏷️ 检测所有类别")
    
    # 运行追踪
    print(f"🔄 开始追踪...")
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",  # 使用 ByteTrack
        conf=confidence,
        classes=class_ids,
        device=device,
        stream=True,  # 流式处理，节省内存
        verbose=False
    )
    
    # 收集追踪结果
    all_tracks = defaultdict(lambda: {"class": None, "frames": {}})
    
    # 准备输出视频
    video_output_path = None
    video_writer = None
    temp_video_path = None
    
    if generate_video:
        video_output_path = output_json.replace('.json', '_annotated.mp4')
        temp_video_path = output_json.replace('.json', '_temp.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    for result in results:
        # 获取当前帧
        frame = result.orig_img.copy()
        
        # 处理检测结果
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # 获取边界框
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # 获取类别
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = model.names[cls_id]
                
                # 获取置信度
                conf = float(boxes.conf[i].cpu().numpy())
                
                # 获取追踪 ID
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                else:
                    track_id = i  # 如果没有追踪 ID，使用索引
                
                # 过滤第一人称视角的车辆（骑行者自己的摩托车）
                if is_first_person_vehicle((x1, y1, x2, y2), height, width, cls_name):
                    continue  # 跳过不标注
                
                # 保存追踪数据
                track_key = f"{cls_name}_{track_id}"
                all_tracks[track_key]["class"] = cls_name
                all_tracks[track_key]["track_id"] = track_id
                all_tracks[track_key]["frames"][frame_idx] = {
                    "x": float(x1) / width * 100,
                    "y": float(y1) / height * 100,
                    "width": float(x2 - x1) / width * 100,
                    "height": float(y2 - y1) / height * 100,
                    "confidence": conf,
                    "time": frame_idx / fps
                }
                
                # 在帧上绘制 - 颜色按类别固定
                if video_writer is not None:
                    color = COLORS.get(cls_name, (0, 255, 0))
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # 绘制标签（显示类别和 ID）
                    label = f"{cls_name} #{track_id}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    cv2.rectangle(frame, (int(x1), int(y1) - th - 6), 
                                  (int(x1) + tw + 6, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1) + 3, int(y1) - 3), 
                                font, font_scale, (255, 255, 255), thickness)
        
        # 写入帧号
        if video_writer is not None:
            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            video_writer.write(frame)
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"   已处理 {frame_idx}/{frame_count} 帧...")
    
    print(f"   处理完成，共 {frame_idx} 帧")
    
    # 关闭视频写入
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
    for track_key, track_data in all_tracks.items():
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
                    "labels": [track_data["class"]]
                },
                "id": track_key
            })
    
    # 保存 JSON
    output_data = [{
        "data": {
            "video": f"/data/local-files/?d={os.path.basename(video_path)}"
        },
        "predictions": [{
            "result": ls_results,
            "model_version": f"YOLO-{model_name}-ByteTrack"
        }]
    }]
    
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 统计
    class_counts = defaultdict(int)
    for track_key, track_data in all_tracks.items():
        class_counts[track_data["class"]] += 1
    
    print(f"\n✅ JSON 结果已保存到: {output_json}")
    print(f"   共追踪到 {len(all_tracks)} 个目标:")
    for cls, count in sorted(class_counts.items()):
        print(f"      - {cls}: {count} 个")
    
    if video_output_path:
        print(f"✅ 标注视频已保存到: {video_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO + ByteTrack 视频目标追踪"
    )
    parser.add_argument(
        "video_path",
        help="输入视频路径"
    )
    parser.add_argument(
        "--output", "-o",
        default="YOLO_output/tracking_result.json",
        help="输出 JSON 路径"
    )
    parser.add_argument(
        "--classes", "-c",
        nargs="+",
        default=["person", "car", "motorcycle", "bicycle", "bus", "truck", "traffic light", "stop sign"],
        help="要检测的类别"
    )
    parser.add_argument(
        "--model", "-m",
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLO 模型 (n=最快, x=最准)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="置信度阈值"
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="计算设备"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="不生成标注视频"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ 视频文件不存在: {args.video_path}")
        sys.exit(1)
    
    run_yolo_tracking(
        video_path=args.video_path,
        output_json=args.output,
        target_classes=args.classes,
        model_name=args.model,
        confidence=args.confidence,
        generate_video=not args.no_video,
        device=args.device
    )


if __name__ == "__main__":
    main()

