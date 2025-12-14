#!/usr/bin/env python3
"""
交通标志检测结果可视化
支持检测框、追踪ID、类别标签的显示

用法:
    python3 scripts/visualize_traffic_signs.py \
        --video data/D1_video_clips/your_video.mp4 \
        --detections traffic_signs_results.json \
        --output traffic_signs_annotated.mp4
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficSignVisualizer:
    """交通标志检测结果可视化器"""

    def __init__(self):
        # 颜色映射 (BGR格式)
        self.colors = {
            'warning': (0, 255, 255),      # 黄色
            'regulatory': (0, 0, 255),     # 红色
            'information': (255, 0, 0),    # 蓝色
            'distance': (0, 255, 0),       # 绿色
            'default': (128, 128, 128)     # 灰色
        }

        # 类别颜色缓存
        self.class_colors = {}

    def _get_class_color(self, sign_class: str) -> Tuple[int, int, int]:
        """获取类别对应的颜色"""
        # 基于类别名称分配颜色
        class_lower = sign_class.lower()

        if any(keyword in class_lower for keyword in ['warning', 'ahead', 'bend', 'cross']):
            return self.colors['warning']
        elif any(keyword in class_lower for keyword in ['stop', 'no_', 'limit', 'must']):
            return self.colors['regulatory']
        elif any(keyword in class_lower for keyword in ['lane', 'route', 'census']):
            return self.colors['information']
        elif 'distance' in class_lower or 'm_' in class_lower:
            return self.colors['distance']
        else:
            # 为其他类别分配随机颜色
            if sign_class not in self.class_colors:
                self.class_colors[sign_class] = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
            return self.class_colors[sign_class]

    def _get_short_class_name(self, sign_class: str, max_length: int = 20) -> str:
        """获取简短的类别名称用于显示"""
        # 移除常见的前缀和后缀
        name = sign_class.replace('_', ' ')

        # 截断过长的名称
        if len(name) > max_length:
            name = name[:max_length-3] + '...'

        return name

    def visualize_detections(self, video_path: str, detections_file: str,
                           output_path: str, show_confidence: bool = True,
                           show_track_id: bool = True, font_scale: float = 0.6):
        """
        可视化检测结果

        Args:
            video_path: 输入视频路径
            detections_file: 检测结果JSON文件
            output_path: 输出视频路径
            show_confidence: 是否显示置信度
            show_track_id: 是否显示轨迹ID
            font_scale: 字体大小
        """
        logger.info(f"开始可视化检测结果")
        logger.info(f"输入视频: {video_path}")
        logger.info(f"检测结果: {detections_file}")
        logger.info(f"输出视频: {output_path}")

        # 加载检测结果
        try:
            with open(detections_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"无法加载检测结果文件: {e}")
            return

        # 解析检测结果
        if 'detection_results' in data:
            # 集成流水线格式
            detection_results = data['detection_results']
            frame_detections = detection_results.get('frames', {})
        elif 'raw_detections' in data:
            # 检测器格式
            raw_detections = data['raw_detections']
            frame_detections = {}
            for det in raw_detections:
                frame_idx = det['frame']
                if frame_idx not in frame_detections:
                    frame_detections[frame_idx] = []
                frame_detections[frame_idx].append(det)
        else:
            logger.error("无法识别的检测结果格式")
            return

        logger.info(f"加载了 {len(frame_detections)} 帧的检测结果")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"视频信息: {width}x{height}, {fps}FPS, {total_frames}帧")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # 字体设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_thickness = 2

        frame_idx = 0
        processed_frames = 0

        # 处理视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 获取当前帧的检测结果
            current_detections = frame_detections.get(frame_idx, [])

            # 绘制检测结果
            for det in current_detections:
                self._draw_detection(frame, det, font, font_scale, font_thickness,
                                   show_confidence, show_track_id)

            # 添加帧信息
            info_text = f"Frame: {frame_idx}/{total_frames} | Detections: {len(current_detections)}"
            cv2.putText(frame, info_text, (10, 30), font, font_scale, (255, 255, 255), font_thickness)

            # 写入输出视频
            out.write(frame)

            processed_frames += 1
            if processed_frames % 100 == 0:
                logger.info(f"已处理 {processed_frames} 帧")

            frame_idx += 1

        # 释放资源
        cap.release()
        out.release()

        logger.info(f"✅ 可视化完成!")
        logger.info(f"处理了 {processed_frames} 帧")
        logger.info(f"输出视频: {output_path}")

    def _draw_detection(self, frame: np.ndarray, detection: Dict[str, Any],
                       font, font_scale: float, font_thickness: int,
                       show_confidence: bool, show_track_id: bool):
        """
        绘制单个检测结果

        Args:
            frame: 视频帧
            detection: 检测结果
            font: 字体
            font_scale: 字体大小
            font_thickness: 字体粗细
            show_confidence: 是否显示置信度
            show_track_id: 是否显示轨迹ID
        """
        bbox = detection['bbox']
        confidence = detection['confidence']
        sign_class = detection['class']
        track_id = detection.get('track_id')

        x, y, w, h = bbox

        # 获取颜色
        color = self._get_class_color(sign_class)

        # 绘制边界框
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # 准备标签文本
        short_name = self._get_short_class_name(sign_class)
        label_parts = [short_name]

        if show_confidence:
            label_parts.append(f"{confidence:.2f}")

        if show_track_id and track_id is not None:
            label_parts.append(f"ID:{track_id}")

        label_text = " ".join(label_parts)

        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # 绘制标签背景
        label_y = max(y, text_height + 10)
        cv2.rectangle(frame, (x, label_y - text_height - baseline - 5),
                     (x + text_width, label_y + baseline - 5), color, -1)

        # 绘制文本
        cv2.putText(frame, label_text, (x, label_y - baseline),
                   font, font_scale, (255, 255, 255), font_thickness)

    def create_detection_summary(self, detections_file: str, output_path: str):
        """
        创建检测结果摘要

        Args:
            detections_file: 检测结果文件
            output_path: 摘要图像输出路径
        """
        logger.info(f"创建检测结果摘要: {detections_file}")

        try:
            with open(detections_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"无法加载检测结果文件: {e}")
            return

        # 统计信息
        if 'raw_detections' in data:
            detections = data['raw_detections']
        else:
            logger.error("无法找到检测结果数据")
            return

        # 按类别统计
        class_counts = {}
        for det in detections:
            sign_class = det['class']
            class_counts[sign_class] = class_counts.get(sign_class, 0) + 1

        if not class_counts:
            logger.warning("没有找到检测结果")
            return

        # 创建摘要图像
        fig_height = max(600, len(class_counts) * 30 + 100)
        summary_img = np.ones((fig_height, 800, 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        # 标题
        title = "Traffic Sign Detection Summary"
        cv2.putText(summary_img, title, (200, 40), font, font_scale * 1.5, (0, 0, 0), font_thickness + 1)

        # 统计信息
        total_detections = len(detections)
        unique_classes = len(class_counts)
        info_text = f"Total Detections: {total_detections} | Unique Classes: {unique_classes}"
        cv2.putText(summary_img, info_text, (200, 80), font, font_scale, (0, 0, 0), font_thickness)

        # 类别统计
        y_offset = 120
        for i, (sign_class, count) in enumerate(sorted(class_counts.items(), key=lambda x: x[1], reverse=True)):
            color = self._get_class_color(sign_class)
            short_name = self._get_short_class_name(sign_class, 35)

            # 类别名称
            text = f"{short_name}: {count}"
            cv2.putText(summary_img, text, (50, y_offset), font, font_scale, color, font_thickness)

            # 绘制小矩形作为示例
            cv2.rectangle(summary_img, (750, y_offset - 15), (780, y_offset + 5), color, -1)

            y_offset += 30

        # 保存摘要图像
        cv2.imwrite(output_path, summary_img)
        logger.info(f"摘要图像保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='交通标志检测结果可视化')
    parser.add_argument('--video', required=True, help='输入视频文件路径')
    parser.add_argument('--detections', required=True, help='检测结果JSON文件路径')
    parser.add_argument('--output', required=True, help='输出视频文件路径')
    parser.add_argument('--summary', help='摘要图像输出路径 (可选)')
    parser.add_argument('--no-confidence', action='store_true', help='不显示置信度')
    parser.add_argument('--no-track-id', action='store_true', help='不显示轨迹ID')
    parser.add_argument('--font-scale', type=float, default=0.6, help='字体大小')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 创建可视化器
        visualizer = TrafficSignVisualizer()

        # 可视化检测结果
        visualizer.visualize_detections(
            args.video, args.detections, args.output,
            show_confidence=not args.no_confidence,
            show_track_id=not args.no_track_id,
            font_scale=args.font_scale
        )

        # 创建摘要图像 (如果指定)
        if args.summary:
            visualizer.create_detection_summary(args.detections, args.summary)

        print(f"\n🎯 可视化完成!")
        print(f"输出视频: {args.output}")
        if args.summary:
            print(f"摘要图像: {args.summary}")

    except Exception as e:
        logger.error(f"可视化失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()