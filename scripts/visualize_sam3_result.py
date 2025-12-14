#!/usr/bin/env python3
"""
可视化 SAM3 追踪结果
将 JSON 标注结果叠加到视频上，生成带边界框的视频
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from collections import defaultdict

# 颜色列表（BGR格式）
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


def load_annotations(json_path: str) -> dict:
    """加载 JSON 标注文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 解析标注结果
    annotations = defaultdict(list)  # {frame_idx: [(box, label, obj_id, color), ...]}
    
    if isinstance(data, list) and len(data) > 0:
        predictions = data[0].get("predictions", [])
        if predictions:
            results = predictions[0].get("result", [])
            
            for idx, result in enumerate(results):
                obj_id = result.get("id", f"obj_{idx}")
                value = result.get("value", {})
                labels = value.get("labels", ["object"])
                label = labels[0] if labels else "object"
                sequence = value.get("sequence", [])
                
                color = COLORS[idx % len(COLORS)]
                
                for frame_data in sequence:
                    frame_idx = frame_data.get("frame", 0)
                    x = frame_data.get("x", 0)
                    y = frame_data.get("y", 0)
                    width = frame_data.get("width", 0)
                    height = frame_data.get("height", 0)
                    
                    annotations[frame_idx].append({
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height,
                        "label": label,
                        "obj_id": obj_id,
                        "color": color
                    })
    
    return annotations


def draw_box(frame, box_data, frame_width, frame_height):
    """在帧上绘制边界框"""
    # 坐标是百分比 (0-100)，转换为像素
    x_pct = box_data["x"]
    y_pct = box_data["y"]
    w_pct = box_data["width"]
    h_pct = box_data["height"]
    
    # 检查坐标是否合理（应该在 0-100 范围内）
    # 如果坐标值太大，可能是像素坐标，需要归一化
    if x_pct > 100 or y_pct > 100 or w_pct > 100 or h_pct > 100:
        # 假设是像素坐标，直接使用
        x1 = int(x_pct)
        y1 = int(y_pct)
        x2 = int(x_pct + w_pct)
        y2 = int(y_pct + h_pct)
        
        # 但如果还是太大，可能需要缩放
        if x2 > frame_width * 10 or y2 > frame_height * 10:
            # 可能是乘以了图像尺寸，需要除回去
            scale_x = frame_width
            scale_y = frame_height
            x1 = int(x_pct / scale_x * frame_width / 100)
            y1 = int(y_pct / scale_y * frame_height / 100)
            x2 = int((x_pct + w_pct) / scale_x * frame_width / 100)
            y2 = int((y_pct + h_pct) / scale_y * frame_height / 100)
    else:
        # 正常的百分比坐标
        x1 = int(x_pct / 100 * frame_width)
        y1 = int(y_pct / 100 * frame_height)
        x2 = int((x_pct + w_pct) / 100 * frame_width)
        y2 = int((y_pct + h_pct) / 100 * frame_height)
    
    # 确保坐标在有效范围内
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))
    
    color = box_data["color"]
    label = box_data["label"]
    obj_id = box_data["obj_id"]
    
    # 绘制边界框
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 绘制标签背景
    text = f"{label} ({obj_id})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
    cv2.putText(frame, text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return frame


def visualize_video(video_path: str, json_path: str, output_path: str):
    """生成带标注的视频"""
    print(f"📹 加载视频: {video_path}")
    print(f"📝 加载标注: {json_path}")
    
    # 加载标注
    annotations = load_annotations(json_path)
    print(f"   共有 {len(annotations)} 帧有标注")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"   视频信息: {frame_width}x{frame_height}, {fps}fps, {frame_count}帧")
    
    # 创建输出视频
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    annotated_frames = 0
    
    print(f"🎬 生成标注视频...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 如果当前帧有标注，绘制边界框
        if frame_idx in annotations:
            for box_data in annotations[frame_idx]:
                frame = draw_box(frame, box_data, frame_width, frame_height)
            annotated_frames += 1
        
        # 在左上角显示帧号
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"   已处理 {frame_idx}/{frame_count} 帧...")
    
    cap.release()
    out.release()
    
    print(f"✅ 标注视频已保存到: {output_path}")
    print(f"   共标注了 {annotated_frames} 帧")


def main():
    parser = argparse.ArgumentParser(description="可视化 SAM3 追踪结果")
    parser.add_argument("video_path", help="原始视频路径")
    parser.add_argument("json_path", help="JSON 标注文件路径")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出视频路径 (默认: 在 JSON 同目录下生成)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ 视频文件不存在: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(args.json_path):
        print(f"❌ JSON 文件不存在: {args.json_path}")
        sys.exit(1)
    
    if args.output is None:
        # 默认输出路径
        json_dir = os.path.dirname(args.json_path)
        json_name = os.path.splitext(os.path.basename(args.json_path))[0]
        args.output = os.path.join(json_dir, f"{json_name}_annotated.mp4")
    
    visualize_video(args.video_path, args.json_path, args.output)


if __name__ == "__main__":
    main()








