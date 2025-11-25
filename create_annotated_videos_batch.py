#!/usr/bin/env python3
"""
批量创建带标注框的可视化视频
"""

import cv2
import json
import os
from pathlib import Path
import sys
import numpy as np

def create_annotated_video(video_path, json_path, output_path):
    """创建带标注框的视频"""
    
    # 读取标注
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data[0]['predictions'][0]['result']
    
    # 按帧号组织（不做插值）
    frame_annotations = {}
    for result in results:
        frame_num = result['value']['frame']
        if frame_num not in frame_annotations:
            frame_annotations[frame_num] = []
        frame_annotations[frame_num].append(result)
    
    # 中英文映射
    category_mapping = {
        "汽车": "Car",
        "交通标志": "Traffic Sign",
        "摩托车": "Motorcycle",
        "行人": "Pedestrian",
        "自行车": "Bicycle",
        "施工区域": "Construction",
    }
    
    # 颜色
    colors = {
        "Car": (66, 165, 245),           # 蓝色
        "Traffic Sign": (156, 39, 176),   # 紫色
        "Motorcycle": (102, 187, 106),    # 绿色
        "Pedestrian": (255, 112, 67),     # 橙色
        "Bicycle": (255, 193, 7),         # 黄色
        "Construction": (255, 87, 34),    # 深橙色
    }
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    labeled_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 只绘制有AI检测的帧
        if frame_count in frame_annotations:
            labeled_frames += 1
            
            for result in frame_annotations[frame_count]:
                value = result['value']
                
                # 1. 处理矩形框
                if 'rectanglelabels' in value:
                    category_cn = value['rectanglelabels'][0]
                    category_en = category_mapping.get(category_cn, category_cn)
                    
                    x = int(value['x'] * width / 100)
                    y = int(value['y'] * height / 100)
                    w = int(value['width'] * width / 100)
                    h = int(value['height'] * height / 100)
                    
                    color = colors.get(category_en, (255, 255, 255))
                    
                    # 绘制矩形框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # 绘制标签
                    label = category_en
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x, y - label_h - 8), (x + label_w + 5, y), color, -1)
                    cv2.putText(frame, label, (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 2. 处理多边形 (SAM3)
                elif 'polygonlabels' in value:
                    category_cn = value['polygonlabels'][0]
                    category_en = category_mapping.get(category_cn, category_cn)
                    points = value['points'] # [[x1, y1], [x2, y2], ...] (0-100)
                    
                    # 转换坐标
                    pts = []
                    for p in points:
                        px = int(p[0] * width / 100)
                        py = int(p[1] * height / 100)
                        pts.append([px, py])
                    
                    pts = np.array(pts, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    color = colors.get(category_en, (255, 255, 255))
                    
                    # 绘制多边形轮廓
                    cv2.polylines(frame, [pts], True, color, 2)
                    
                    # 绘制半透明填充
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # 状态信息（绿色）
            num_objs = len(frame_annotations[frame_count])
            info_text = f"Frame: {frame_count}/{total_frames} | AI | Objects: {num_objs}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # 无标注帧（灰色提示）
            info_text = f"Frame: {frame_count}/{total_frames}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # SR:5标记
        cv2.putText(frame, "SR:5", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    return True


def main():
    """批量处理"""
    
    print("=" * 70)
    print("🎬 批量创建带标注框的可视化视频")
    print("=" * 70)
    print()
    
    json_dir = "labels/batch_output/json"
    video_dir = "data/D1_video_clips"
    output_dir = "labels/batch_output/videos"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = sorted(Path(json_dir).glob("*_sr5.json"))
    
    total = len(json_files)
    success = 0
    failed = 0
    
    print(f"📊 找到 {total} 个标注文件")
    print(f"📁 输出目录: {output_dir}")
    print()
    
    for i, json_path in enumerate(json_files, 1):
        # 提取视频文件名
        basename = json_path.stem.replace("_sr5", "")
        video_path = os.path.join(video_dir, f"{basename}.mp4")
        output_path = os.path.join(output_dir, f"{basename}_annotated.mp4")
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[{i}/{total}] {basename}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if not os.path.exists(video_path):
            print(f"❌ 视频文件不存在: {video_path}")
            failed += 1
            continue
        
        if os.path.exists(output_path):
            print(f"⏭️  跳过（已存在）: {output_path}")
            success += 1
            continue
        
        print(f"   视频: {video_path}")
        print(f"   标注: {json_path}")
        print(f"   输出: {output_path}")
        print()
        
        # 创建视频
        if create_annotated_video(str(video_path), str(json_path), output_path):
            file_size = Path(output_path).stat().st_size / 1024 / 1024
            print(f"   ✅ 成功！文件大小: {file_size:.2f} MB")
            success += 1
        else:
            print(f"   ❌ 失败")
            failed += 1
        
        print()
    
    print("=" * 70)
    print("✅ 批量创建完成！")
    print("=" * 70)
    print()
    print(f"📊 统计:")
    print(f"  - 总数: {total}")
    print(f"  - 成功: {success}")
    print(f"  - 失败: {failed}")
    print()
    print(f"📁 可视化视频位置: {output_dir}/")
    print()
    print("🎬 查看视频:")
    print(f"   open {output_dir}")
    print()


if __name__ == "__main__":
    main()


