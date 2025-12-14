#!/usr/bin/env python3
"""
交通标志检测器 - 基于模板匹配的交通标志检测系统
将signs数据集集成到SAM3工作流中

用法:
    python3 scripts/traffic_sign_detector.py \
        --video data/D1_video_clips/your_video.mp4 \
        --output traffic_signs_detections.json \
        --signs-dir signs/highres/png2560px/
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficSignDetector:
    """交通标志检测器"""

    def __init__(self, signs_dir: str, threshold: float = 0.7):
        """
        初始化交通标志检测器

        Args:
            signs_dir: 交通标志图像目录
            threshold: 模板匹配阈值
        """
        self.signs_dir = Path(signs_dir)
        self.threshold = threshold
        self.sign_templates = {}
        self.sign_classes = {}

        # 加载所有交通标志模板
        self._load_sign_templates()

    def _load_sign_templates(self):
        """加载交通标志模板"""
        logger.info(f"正在加载交通标志模板从 {self.signs_dir}")

        if not self.signs_dir.exists():
            raise FileNotFoundError(f"交通标志目录不存在: {self.signs_dir}")

        # 支持的图像格式
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

        for img_path in self.signs_dir.glob('*'):
            if img_path.suffix.lower() in image_extensions:
                try:
                    # 读取图像
                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img is None:
                        logger.warning(f"无法读取图像: {img_path}")
                        continue

                    # 获取标志名称（去除文件扩展名）
                    sign_name = img_path.stem

                    # 存储模板信息
                    self.sign_templates[sign_name] = {
                        'image': img,
                        'gray': cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        'size': img.shape[:2],
                        'filename': img_path.name
                    }

                    logger.info(f"加载模板: {sign_name} ({img.shape})")

                except Exception as e:
                    logger.error(f"加载模板失败 {img_path}: {e}")

        logger.info(f"成功加载 {len(self.sign_templates)} 个交通标志模板")

    def _multi_scale_template_match(self, frame: np.ndarray, template: np.ndarray,
                                   sign_name: str) -> List[Dict[str, Any]]:
        """
        多尺度模板匹配

        Args:
            frame: 输入帧
            template: 模板图像
            sign_name: 标志名称

        Returns:
            检测结果列表
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        template_h, template_w = template.shape[:2]
        detections = []

        # 多尺度范围 (0.2 到 2.0 倍)
        scales = np.linspace(0.2, 2.0, 15)

        for scale in scales:
            # 缩放模板
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)

            if scaled_w > frame.shape[1] or scaled_h > frame.shape[0]:
                continue

            # 缩放模板
            scaled_template = cv2.resize(template, (scaled_w, scaled_h))

            # 模板匹配
            result = cv2.matchTemplate(frame_gray, scaled_template, cv2.TM_CCOEFF_NORMED)

            # 找到匹配位置
            locations = np.where(result >= self.threshold)

            for pt in zip(*locations[::-1]):  # 切换 x和y 坐标
                match_value = result[pt[1], pt[0]]

                detection = {
                    'bbox': [pt[0], pt[1], scaled_w, scaled_h],  # [x, y, w, h]
                    'confidence': float(match_value),
                    'class': sign_name,
                    'scale': scale
                }
                detections.append(detection)

        return detections

    def detect_frame(self, frame: np.ndarray, frame_idx: int = 0) -> List[Dict[str, Any]]:
        """
        检测单帧中的交通标志

        Args:
            frame: 输入帧
            frame_idx: 帧索引

        Returns:
            检测结果列表
        """
        all_detections = []

        # 对每个模板进行匹配
        for sign_name, template_info in self.sign_templates.items():
            detections = self._multi_scale_template_match(
                frame, template_info['gray'], sign_name
            )

            # 添加帧信息
            for detection in detections:
                detection['frame'] = frame_idx
                detection['template_size'] = template_info['size']

            all_detections.extend(detections)

        # 非极大值抑制 (NMS)
        filtered_detections = self._apply_nms(all_detections)

        return filtered_detections

    def _apply_nms(self, detections: List[Dict[str, Any]],
                   nms_threshold: float = 0.4) -> List[Dict[str, Any]]:
        """
        应用非极大值抑制去除重叠检测

        Args:
            detections: 检测结果列表
            nms_threshold: NMS阈值

        Returns:
            过滤后的检测结果
        """
        if not detections:
            return []

        # 转换格式用于NMS
        boxes = []
        scores = []
        classes = []

        for detection in detections:
            x, y, w, h = detection['bbox']
            boxes.append([x, y, x + w, y + h])
            scores.append(detection['confidence'])
            classes.append(detection['class'])

        boxes = np.array(boxes)
        scores = np.array(scores)

        # 应用NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.threshold, nms_threshold
        )

        # 过滤检测结果
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_detections = [detections[i] for i in indices]
        else:
            filtered_detections = []

        return filtered_detections

    def detect_video(self, video_path: str, output_path: str,
                    sample_rate: int = 5) -> Dict[str, Any]:
        """
        检测视频中的交通标志

        Args:
            video_path: 视频文件路径
            output_path: 输出JSON文件路径
            sample_rate: 采样率 (每N帧处理一次)

        Returns:
            检测结果
        """
        logger.info(f"开始处理视频: {video_path}")

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"视频信息: {frame_count}帧, {fps}FPS, {width}x{height}")

        all_detections = []
        frame_idx = 0

        # 处理视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 采样处理
            if frame_idx % sample_rate == 0:
                detections = self.detect_frame(frame, frame_idx)
                all_detections.extend(detections)

                if frame_idx % 50 == 0:
                    logger.info(f"已处理 {frame_idx}/{frame_count} 帧, 检测到 {len(detections)} 个标志")

            frame_idx += 1

        cap.release()
        logger.info(f"视频处理完成, 共检测到 {len(all_detections)} 个交通标志")

        # 转换为SAM3格式
        sam3_results = self._convert_to_sam3_format(
            all_detections, video_path, fps, frame_count
        )

        # 保存结果
        self._save_results(sam3_results, output_path)

        return sam3_results

    def _convert_to_sam3_format(self, detections: List[Dict[str, Any]],
                               video_path: str, fps: float, frame_count: int) -> Dict[str, Any]:
        """
        将检测结果转换为SAM3/LabRL Studio格式

        Args:
            detections: 检测结果列表
            video_path: 视频路径
            fps: 帧率
            frame_count: 总帧数

        Returns:
            SAM3格式的结果
        """
        # 按帧分组检测
        frame_detections = {}
        for detection in detections:
            frame_idx = detection['frame']
            if frame_idx not in frame_detections:
                frame_detections[frame_idx] = []
            frame_detections[frame_idx].append(detection)

        # 创建检测结果JSON (兼容现有SAM3格式)
        sam3_data = []

        # 简化的SAM3格式检测结果
        detection_results = {
            "video_info": {
                "path": video_path,
                "fps": fps,
                "frame_count": frame_count,
                "total_detections": len(detections)
            },
            "frames": {}
        }

        # 按帧组织检测结果
        for frame_idx, frame_dets in frame_detections.items():
            frame_time = frame_idx / fps if fps > 0 else frame_idx

            detection_results["frames"][frame_idx] = {
                "timestamp": frame_time,
                "detections": []
            }

            for det in frame_dets:
                x, y, w, h = det['bbox']

                # 转换为相对坐标 (百分比)
                rel_x = (x / 100)  # 假设输入视频是100单位宽，需要调整
                rel_y = (y / 100)  # 这里需要根据实际视频尺寸调整
                rel_width = (w / 100)
                rel_height = (h / 100)

                detection_entry = {
                    "bbox": det['bbox'],  # 原始像素坐标
                    "confidence": det['confidence'],
                    "class": det['class'],
                    "frame": frame_idx,
                    "time": frame_time
                }

                detection_results["frames"][frame_idx]["detections"].append(detection_entry)

        # 创建Label Studio兼容格式
        label_studio_format = [{
            "data": {
                "video": f"/data/local-files/?d={Path(video_path).name}"
            },
            "predictions": [{
                "result": [],
                "score": 0.0
            }]
        }]

        # 这里可以进一步添加轨迹处理逻辑
        # 目前先保存原始检测结果

        return {
            "detection_results": detection_results,
            "label_studio_format": label_studio_format,
            "raw_detections": detections
        }

    def _save_results(self, results: Dict[str, Any], output_path: str):
        """保存检测结果"""
        logger.info(f"保存结果到: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("结果保存完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='交通标志检测器')
    parser.add_argument('--video', required=True, help='输入视频文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--signs-dir', default='signs/highres/png2560px/',
                       help='交通标志图像目录')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='模板匹配阈值')
    parser.add_argument('--sample-rate', type=int, default=5,
                       help='采样率 (每N帧处理一次)')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 创建检测器
        detector = TrafficSignDetector(args.signs_dir, args.threshold)

        # 检测视频
        results = detector.detect_video(
            args.video, args.output, args.sample_rate
        )

        # 打印统计信息
        detection_results = results["detection_results"]
        total_detections = detection_results["video_info"]["total_detections"]
        frames_with_detections = len(detection_results["frames"])

        print(f"\n🎯 检测完成!")
        print(f"总检测数: {total_detections}")
        print(f"有检测的帧数: {frames_with_detections}")
        print(f"结果保存到: {args.output}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()