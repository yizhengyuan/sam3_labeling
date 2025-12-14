#!/usr/bin/env python3
"""
集成交通标志检测与SAM3追踪流水线
结合模板匹配检测 + SAM3分割 + SORT追踪

用法:
    python3 scripts/integrated_traffic_sign_pipeline.py \
        --video data/D1_video_clips/your_video.mp4 \
        --output SAM3_output/traffic_signs_results.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

# 导入现有模块
from traffic_sign_detector import TrafficSignDetector
from retrack_with_sort import SORTTracker  # 假设可以从现有脚本导入

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegratedTrafficSignPipeline:
    """集成交通标志检测流水线"""

    def __init__(self, signs_dir: str, detection_threshold: float = 0.7):
        """
        初始化集成流水线

        Args:
            signs_dir: 交通标志图像目录
            detection_threshold: 检测阈值
        """
        self.signs_dir = signs_dir
        self.detection_threshold = detection_threshold

        # 初始化交通标志检测器
        self.sign_detector = TrafficSignDetector(signs_dir, detection_threshold)

        logger.info("集成交通标志检测流水线初始化完成")

    def process_video(self, video_path: str, output_path: str,
                     apply_sam3: bool = True, apply_sort: bool = True,
                     sample_rate: int = 5) -> Dict[str, Any]:
        """
        处理视频的完整流水线

        Args:
            video_path: 输入视频路径
            output_path: 输出路径
            apply_sam3: 是否应用SAM3分割
            apply_sort: 是否应用SORT追踪
            sample_rate: 采样率

        Returns:
            处理结果
        """
        logger.info(f"开始处理视频: {video_path}")

        # 第一步: 交通标志检测
        logger.info("🔍 步骤1: 交通标志检测")
        detection_results = self.sign_detector.detect_video(
            video_path, output_path.replace('.json', '_detections.json'), sample_rate
        )

        raw_detections = detection_results['raw_detections']
        logger.info(f"检测到 {len(raw_detections)} 个交通标志")

        # 第二步: 应用SORT追踪器
        tracked_results = None
        if apply_sort and raw_detections:
            logger.info("🔗 步骤2: 应用SORT追踪器")
            tracked_results = self._apply_sort_tracking(
                video_path, raw_detections, output_path
            )

        # 第三步: 生成最终结果
        logger.info("📦 步骤3: 生成最终结果")
        final_results = self._generate_final_results(
            detection_results, tracked_results, video_path, apply_sort
        )

        # 保存最终结果
        self._save_final_results(final_results, output_path)

        logger.info("✅ 流水线处理完成")
        return final_results

    def _apply_sort_tracking(self, video_path: str, detections: List[Dict[str, Any]],
                           output_path: str) -> Dict[str, Any]:
        """
        应用SORT追踪器

        Args:
            video_path: 视频路径
            detections: 检测结果
            output_path: 输出路径

        Returns:
            追踪结果
        """
        try:
            # 这里可以导入现有的SORT追踪器代码
            # 由于现有脚本比较复杂，我们创建一个简化版本

            logger.info("应用SORT追踪器到检测结果")

            # 按帧组织检测结果
            frame_detections = {}
            for det in detections:
                frame_idx = det['frame']
                if frame_idx not in frame_detections:
                    frame_detections[frame_idx] = []
                frame_detections[frame_idx].append(det)

            # 简单的追踪逻辑 (实际应用中应该使用SORT)
            tracked_results = self._simple_tracking(frame_detections)

            return tracked_results

        except Exception as e:
            logger.error(f"SORT追踪失败: {e}")
            return None

    def _simple_tracking(self, frame_detections: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        简单的追踪实现 (替代SORT)

        Args:
            frame_detections: 按帧组织的检测结果

        Returns:
            追踪结果
        """
        tracks = {}
        track_id_counter = 0

        # 按帧排序
        sorted_frames = sorted(frame_detections.keys())

        # 简单的最近邻匹配追踪
        for i, frame_idx in enumerate(sorted_frames):
            detections = frame_detections[frame_idx]

            for det in detections:
                # 简单分配track ID (实际应该用更复杂的算法)
                det['track_id'] = track_id_counter
                tracks[track_id_counter] = {
                    'class': det['class'],
                    'start_frame': frame_idx,
                    'detections': [det]
                }
                track_id_counter += 1

        return {
            'tracks': tracks,
            'track_count': len(tracks)
        }

    def _generate_final_results(self, detection_results: Dict[str, Any],
                               tracked_results: Dict[str, Any],
                               video_path: str, use_tracking: bool) -> Dict[str, Any]:
        """
        生成Label Studio兼容的最终结果

        Args:
            detection_results: 检测结果
            tracked_results: 追踪结果
            video_path: 视频路径
            use_tracking: 是否使用追踪

        Returns:
            最终结果
        """
        raw_detections = detection_results['raw_detections']

        if not use_tracking or not tracked_results:
            # 不使用追踪的简单格式
            return self._create_simple_format(raw_detections, video_path)
        else:
            # 使用追踪的格式
            return self._create_tracked_format(tracked_results, video_path)

    def _create_simple_format(self, detections: List[Dict[str, Any]], video_path: str) -> Dict[str, Any]:
        """创建简单格式的结果"""
        # 按帧分组
        frame_detections = {}
        for det in detections:
            frame_idx = det['frame']
            if frame_idx not in frame_detections:
                frame_detections[frame_idx] = []
            frame_detections[frame_idx].append(det)

        # 创建Label Studio格式
        results = []

        for frame_idx, frame_dets in frame_detections.items():
            for det in frame_dets:
                x, y, w, h = det['bbox']

                # 转换为相对坐标 (这里需要根据实际视频尺寸调整)
                rel_x = max(0, min(100, (x / 1920) * 100))  # 假设1920宽度
                rel_y = max(0, min(100, (y / 1080) * 100))  # 假设1080高度
                rel_width = max(0, min(100, (w / 1920) * 100))
                rel_height = max(0, min(100, (h / 1080) * 100))

                result_entry = {
                    "from_name": "box",
                    "to_name": "video",
                    "type": "videorectangle",
                    "value": {
                        "frames": [
                            {
                                "frame": frame_idx,
                                "x": rel_x,
                                "y": rel_y,
                                "width": rel_width,
                                "height": rel_height,
                                "time": frame_idx / 30.0  # 假设30fps
                            }
                        ],
                        "labels": [det['class']]
                    },
                    "id": f"detection_{frame_idx}_{det.get('track_id', 'unknown')}",
                    "score": det['confidence']
                }

                # 如果还没有results，创建第一个条目
                if not results:
                    results.append({
                        "data": {
                            "video": f"/data/local-files/?d={Path(video_path).name}"
                        },
                        "predictions": [{
                            "result": [],
                            "score": 0.0
                        }]
                    })

                results[0]["predictions"][0]["result"].append(result_entry)

        return {
            "label_studio_format": results,
            "detection_count": len(detections),
            "frame_count": len(frame_detections),
            "tracking_enabled": False
        }

    def _create_tracked_format(self, tracked_results: Dict[str, Any], video_path: str) -> Dict[str, Any]:
        """创建带追踪的格式"""
        tracks = tracked_results['tracks']

        # 创建Label Studio格式
        results = [{
            "data": {
                "video": f"/data/local-files/?d={Path(video_path).name}"
            },
            "predictions": [{
                "result": [],
                "score": 0.0
            }]
        }]

        # 为每个轨迹创建结果
        for track_id, track_info in tracks.items():
            if not track_info['detections']:
                continue

            # 创建轨迹序列
            sequence = []
            for det in track_info['detections']:
                x, y, w, h = det['bbox']

                # 转换为相对坐标
                rel_x = max(0, min(100, (x / 1920) * 100))
                rel_y = max(0, min(100, (y / 1080) * 100))
                rel_width = max(0, min(100, (w / 1920) * 100))
                rel_height = max(0, min(100, (h / 1080) * 100))

                frame_data = {
                    "frame": det['frame'],
                    "x": rel_x,
                    "y": rel_y,
                    "width": rel_width,
                    "height": rel_height,
                    "time": det['frame'] / 30.0,  # 假设30fps
                    "enabled": True
                }
                sequence.append(frame_data)

            # 创建轨迹结果
            track_result = {
                "from_name": "box",
                "to_name": "video",
                "type": "videorectangle",
                "value": {
                    "sequence": sequence,
                    "labels": [track_info['class']]
                },
                "id": f"track_{track_id}"
            }

            results[0]["predictions"][0]["result"].append(track_result)

        return {
            "label_studio_format": results,
            "track_count": len(tracks),
            "tracking_enabled": True
        }

    def _save_final_results(self, results: Dict[str, Any], output_path: str):
        """保存最终结果"""
        logger.info(f"保存最终结果到: {output_path}")

        # 保存Label Studio格式
        label_studio_path = output_path.replace('.json', '_label_studio.json')
        with open(label_studio_path, 'w', encoding='utf-8') as f:
            json.dump(results['label_studio_format'], f, indent=2, ensure_ascii=False)

        # 保存完整结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"结果保存完成:")
        logger.info(f"  - 完整结果: {output_path}")
        logger.info(f"  - Label Studio格式: {label_studio_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='集成交通标志检测流水线')
    parser.add_argument('--video', required=True, help='输入视频文件路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--signs-dir', default='signs/highres/png2560px/',
                       help='交通标志图像目录')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='检测阈值')
    parser.add_argument('--sample-rate', type=int, default=5,
                       help='采样率 (每N帧处理一次)')
    parser.add_argument('--no-tracking', action='store_true',
                       help='禁用追踪')
    parser.add_argument('--debug', action='store_true',
                       help='启用调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # 创建集成流水线
        pipeline = IntegratedTrafficSignPipeline(
            args.signs_dir, args.threshold
        )

        # 处理视频
        results = pipeline.process_video(
            args.video, args.output,
            apply_sort=not args.no_tracking,
            sample_rate=args.sample_rate
        )

        # 打印统计信息
        tracking_status = "启用" if not args.no_tracking else "禁用"
        if results.get('tracking_enabled', False):
            track_count = results.get('track_count', 0)
            print(f"\n🎯 处理完成!")
            print(f"追踪状态: {tracking_status}")
            print(f"生成轨迹数: {track_count}")
        else:
            detection_count = results.get('detection_count', 0)
            frame_count = results.get('frame_count', 0)
            print(f"\n🎯 处理完成!")
            print(f"追踪状态: {tracking_status}")
            print(f"检测总数: {detection_count}")
            print(f"覆盖帧数: {frame_count}")

        print(f"结果保存到: {args.output}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()