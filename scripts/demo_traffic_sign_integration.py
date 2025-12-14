#!/usr/bin/env python3
"""
交通标志检测集成演示脚本
展示完整的使用流程：检测 → 追踪 → 可视化 → Label Studio输出

用法:
    python3 scripts/demo_traffic_sign_integration.py
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    logger.info(f"🔄 {description}")
    logger.info(f"执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ 命令执行成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ 命令执行失败: {e}")
        if e.stdout:
            logger.error(f"标准输出: {e.stdout}")
        if e.stderr:
            logger.error(f"错误输出: {e.stderr}")
        return False

def check_files():
    """检查必要的文件和目录"""
    logger.info("🔍 检查必要文件...")

    required_files = [
        "scripts/traffic_sign_detector.py",
        "scripts/integrated_traffic_sign_pipeline.py",
        "scripts/visualize_traffic_signs.py",
        "config/traffic_sign_config.json",
        "signs/highres/png2560px/"
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error("❌ 缺少以下文件:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False

    # 检查交通标志图像数量
    signs_dir = Path("signs/highres/png2560px/")
    image_files = list(signs_dir.glob("*.png"))
    logger.info(f"📊 找到 {len(image_files)} 个交通标志图像")

    if len(image_files) == 0:
        logger.error("❌ 交通标志目录为空")
        return False

    return True

def find_sample_video():
    """查找示例视频文件"""
    video_dirs = [
        "data/D1_video_clips/",
        "data/",
        "temp_frames/"
    ]

    for video_dir in video_dirs:
        if Path(video_dir).exists():
            video_files = list(Path(video_dir).glob("*.mp4"))
            if video_files:
                return str(video_files[0])  # 返回第一个找到的视频

    return None

def create_sample_detection_json():
    """创建示例检测结果JSON（用于演示）"""
    logger.info("📝 创建示例检测结果...")

    sample_data = {
        "detection_results": {
            "video_info": {
                "path": "sample_video.mp4",
                "fps": 30.0,
                "frame_count": 100,
                "total_detections": 15
            },
            "frames": {
                10: {
                    "timestamp": 0.33,
                    "detections": [
                        {
                            "bbox": [800, 400, 120, 120],
                            "confidence": 0.85,
                            "class": "Stop_and_give_way",
                            "frame": 10,
                            "time": 0.33,
                            "track_id": 0
                        }
                    ]
                },
                25: {
                    "timestamp": 0.83,
                    "detections": [
                        {
                            "bbox": [600, 300, 100, 100],
                            "confidence": 0.78,
                            "class": "Speed_limit_(in_km_h)",
                            "frame": 25,
                            "time": 0.83,
                            "track_id": 1
                        }
                    ]
                }
            }
        },
        "raw_detections": [
            {
                "bbox": [800, 400, 120, 120],
                "confidence": 0.85,
                "class": "Stop_and_give_way",
                "frame": 10,
                "time": 0.33,
                "track_id": 0
            },
            {
                "bbox": [600, 300, 100, 100],
                "confidence": 0.78,
                "class": "Speed_limit_(in_km_h)",
                "frame": 25,
                "time": 0.83,
                "track_id": 1
            }
        ]
    }

    # 确保输出目录存在
    output_dir = Path("SAM3_output")
    output_dir.mkdir(exist_ok=True)

    # 保存示例数据
    output_file = output_dir / "sample_traffic_signs.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ 示例数据已保存到: {output_file}")
    return str(output_file)

def demo_detection_only():
    """演示仅检测功能"""
    logger.info("🎯 演示1: 交通标志检测")

    # 创建示例数据用于演示
    sample_file = create_sample_detection_json()
    logger.info(f"📁 创建了示例检测结果: {sample_file}")

    # 显示检测结果统计
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    detection_count = len(data.get('raw_detections', []))
    frame_count = len(data.get('detection_results', {}).get('frames', {}))

    print(f"\n📊 检测统计:")
    print(f"  - 总检测数: {detection_count}")
    print(f"  - 覆盖帧数: {frame_count}")

    return sample_file

def demo_visualization(detection_file):
    """演示可视化功能"""
    logger.info("🎨 演示2: 结果可视化")

    # 查找示例视频
    sample_video = find_sample_video()
    if not sample_video:
        logger.warning("⚠️  未找到示例视频，跳过可视化演示")
        return None

    # 输出路径
    output_video = "SAM3_output/demo_traffic_signs_annotated.mp4"
    output_summary = "SAM3_output/demo_traffic_signs_summary.png"

    # 可视化命令
    viz_cmd = [
        "python3", "scripts/visualize_traffic_signs.py",
        "--video", sample_video,
        "--detections", detection_file,
        "--output", output_video,
        "--summary", output_summary
    ]

    # 运行可视化
    success = run_command(viz_cmd, "生成可视化结果")

    if success:
        logger.info(f"✅ 可视化完成:")
        logger.info(f"  - 标注视频: {output_video}")
        logger.info(f"  - 检测摘要: {output_summary}")
        return output_video, output_summary
    else:
        return None, None

def demo_integration_workflow():
    """演示完整的集成工作流"""
    logger.info("🔄 演示3: 完整集成工作流")

    # 查找示例视频
    sample_video = find_sample_video()
    if not sample_video:
        logger.warning("⚠️  未找到示例视频，跳过完整工作流演示")
        return None

    # 输出路径
    output_file = "SAM3_output/integrated_workflow_results.json"

    # 完整工作流命令
    workflow_cmd = [
        "python3", "scripts/integrated_traffic_sign_pipeline.py",
        "--video", sample_video,
        "--output", output_file,
        "--signs-dir", "signs/highres/png2560px/",
        "--threshold", "0.7",
        "--sample-rate", "10"  # 演示用，每10帧处理一次
    ]

    # 运行完整工作流
    success = run_command(workflow_cmd, "运行完整集成工作流")

    if success:
        logger.info(f"✅ 集成工作流完成: {output_file}")
        return output_file
    else:
        return None

def show_results_summary(results):
    """显示结果摘要"""
    logger.info("📈 结果摘要")

    for step, result in results.items():
        if result:
            logger.info(f"✅ {step}: 成功")
            if isinstance(result, (list, tuple)):
                for item in result:
                    if item:
                        logger.info(f"   📁 {item}")
            else:
                logger.info(f"   📁 {result}")
        else:
            logger.info(f"❌ {step}: 失败或跳过")

def main():
    """主演示函数"""
    parser = argparse.ArgumentParser(description='交通标志检测集成演示')
    parser.add_argument('--step', choices=['check', 'detect', 'visualize', 'workflow', 'all'],
                       default='all', help='演示特定步骤')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🚦 交通标志检测集成演示")
    print("=" * 50)

    results = {}

    try:
        # 步骤1: 环境检查
        if args.step in ['check', 'all']:
            print("\n🔍 步骤1: 环境检查")
            if check_files():
                results['环境检查'] = "环境就绪"
            else:
                logger.error("❌ 环境检查失败，请安装必要文件")
                return

        # 步骤2: 检测演示
        if args.step in ['detect', 'all']:
            print("\n🎯 步骤2: 交通标志检测演示")
            detection_file = demo_detection_only()
            results['交通标志检测'] = detection_file

        # 步骤3: 可视化演示
        if args.step in ['visualize', 'all'] and args.step != 'detect':
            print("\n🎨 步骤3: 结果可视化演示")
            if 'detection_file' not in locals():
                detection_file = demo_detection_only()

            viz_result = demo_visualization(detection_file)
            if viz_result:
                results['结果可视化'] = viz_result

        # 步骤4: 完整工作流演示
        if args.step in ['workflow', 'all'] and args.step != 'detect' and args.step != 'visualize':
            print("\n🔄 步骤4: 完整集成工作流演示")
            workflow_result = demo_integration_workflow()
            results['完整工作流'] = workflow_result

        # 显示结果摘要
        print("\n📈 演示结果摘要")
        print("=" * 30)
        show_results_summary(results)

        # 下一步建议
        print("\n🎯 下一步建议:")
        print("1. 将您的视频放入 data/ 目录")
        print("2. 运行完整检测流水线:")
        print("   python3 scripts/integrated_traffic_sign_pipeline.py \\")
        print("     --video your_video.mp4 \\")
        print("     --output results.json")
        print("3. 查看详细指南: TRAFFIC_SIGNS_INTEGRATION_GUIDE.md")
        print("4. 将结果导入Label Studio进行人工审核")

        print("\n🎉 演示完成!")

    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()