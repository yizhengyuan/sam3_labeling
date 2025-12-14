# 🎯 SAM3 视频标注完整流程指南

## 📋 概述

本指南演示如何从输入 video clip 到输出高质量标注结果的完整流程，结合 SAM3 检测和 SORT 追踪器优化。

## 🎬 输入要求

### 视频文件
- **格式**: MP4, AVI, MOV 等常见格式
- **分辨率**: 任意 (示例使用 1920x1080)
- **帧率**: 任意 (示例使用 22.5 fps)
- **时长**: 建议 < 30 秒用于测试

### 环境依赖
```bash
# 必需依赖
pip install torch torchvision
pip install opencv-python
pip install numpy
pip install filterpy
pip install scipy
```

## 🚀 完整流程

### 方案1: 使用现有 SAM3 检测数据 (推荐)

#### 步骤1: 准备输入
```bash
# 检查视频文件
ls data/D1_video_clips/D1_rand11-15_clip_000.mp4
```

#### 步骤2: 使用现有检测数据
```bash
# 使用已有的高质量逐帧检测结果
# 文件: SAM3_output/clip_000_every_frame.json
# - 691 个检测框
# - 覆盖 287 帧
# - 检测目标: car
```

#### 步骤3: 应用 SORT 追踪器
```bash
python3 scripts/retrack_with_sort.py SAM3_output/clip_000_every_frame.json \
  --output SAM3_output/workflow_final_results.json \
  --video data/D1_video_clips/D1_rand11-15_clip_000.mp4 \
  --max-age 20 \
  --min-hits 2 \
  --iou-threshold 0.25
```

### 方案2: 从头开始 (需要解决 SAM3 依赖)

#### 安装 SAM3 (可能遇到依赖问题)
```bash
# 由于 Apple Silicon + Python 3.13 兼容性问题，可能无法直接安装
cd sam3_repo
pip install -e .
```

#### 运行原始检测
```bash
python3 scripts/sam3_video_tracking.py your_video.mp4 \
  --text "car" "person" "traffic sign" \
  --sample-rate 1 \
  --output your_detection_results.json
```

## 📊 输出结果

### 文件结构
```
SAM3_output/
├── workflow_final_results.json          # 最终标注结果 (Label Studio 格式)
└── workflow_final_results_annotated.mp4 # 可视化标注视频
```

### JSON 格式 (Label Studio 兼容)
```json
[
  {
    "data": {
      "video": "/data/local-files/?d=your_video.mp4"
    },
    "predictions": [
      {
        "result": [
          {
            "from_name": "box",
            "to_name": "video",
            "type": "videorectangle",
            "value": {
              "sequence": [
                {
                  "frame": 6,
                  "x": 46.2,
                  "y": 32.4,
                  "width": 3.2,
                  "height": 4.9,
                  "time": 0.27,
                  "enabled": true
                }
                // ... 更多帧数据
              ],
              "labels": ["car"]
            },
            "id": "track_0"
          }
          // ... 更多轨迹
        ]
      }
    ]
  }
]
```

## ⚙️ 参数调优

### SORT 追踪器参数
```bash
--max-age 20       # 目标消失后保留 20 帧
--min-hits 2       # 需要连续检测 2 次才建立轨迹
--iou-threshold 0.25 # IoU 匹配阈值
```

- **max-age**: 增大可处理更长遮挡，减小可更快清理轨迹
- **min-hits**: 增大减少噪声，减小提高检测灵敏度
- **iou-threshold**: 增大适应目标快速移动，减小提高匹配精度

### SAM3 检测参数
```bash
--text "car" "person"      # 检测目标列表
--sample-rate 1           # 采样率 (1=逐帧, 5=每5帧)
--confidence 0.3          # 置信度阈值
--device mps             # 计算设备 (mps/cuda/cpu)
```

## 🎯 结果分析

### 处理效果
- **原始检测**: 691 个散乱检测框
- **SORT 优化**: 21 条稳定轨迹
- **最长轨迹**: 285 帧 (99.3% 视频时长)
- **数据压缩**: JSON 文件大小减少 13%

### 质量提升
- ✅ **ID 稳定**: 每个目标保持唯一标识
- ✅ **轨迹连续**: 填补检测间隙
- ✅ **噪声过滤**: 去除短暂误检测
- ✅ **平滑轨迹**: 卡尔曼滤波减少抖动

## 🔧 故障排除

### 常见问题

#### 1. SAM3 依赖问题
```
错误: No module named 'triton'
解决: 使用现有检测数据，或安装兼容版本
```

#### 2. 设备选择
```bash
# Apple Silicon
--device mps

# NVIDIA GPU
--device cuda

# CPU (较慢)
--device cpu
```

#### 3. 内存不足
```bash
# 增加采样率减少处理量
--sample-rate 5  # 从 1 改为 5

# 减少检测类别
--text "car"     # 只检测汽车
```

## 📝 使用示例

### 基础使用
```bash
# 1. 准备视频
cp your_video.mp4 data/D1_video_clips/

# 2. 应用 SORT 追踪器
python3 scripts/retrack_with_sort.py SAM3_output/clip_000_every_frame.json \
  --output SAM3_output/my_results.json \
  --video data/D1_video_clips/your_video.mp4

# 3. 查看结果
open SAM3_output/my_results_annotated.mp4
```

### 高级参数调优
```bash
python3 scripts/retrack_with_sort.py input.json \
  --output output.json \
  --video video.mp4 \
  --max-age 30 \
  --min-hits 3 \
  --iou-threshold 0.3 \
  --debug
```

## 🎉 成功标准

✅ **流程完成**:
- 输入视频 → 检测数据 → SORT 优化 → 标注输出
- 生成 JSON 和标注视频文件
- JSON 格式兼容 Label Studio

✅ **质量指标**:
- 轨迹 ID 稳定
- 长轨迹覆盖主要目标
- 短轨迹被过滤

✅ **文件验证**:
```bash
# 检查输出文件
ls -lh SAM3_output/workflow_final_results*

# 验证 JSON 格式
python3 -m json.tool SAM3_output/workflow_final_results.json > /dev/null && echo "JSON 格式正确"
```

## 📞 支持

如遇到问题:
1. 检查依赖安装
2. 验证视频文件格式
3. 调整参数设置
4. 查看错误日志

---

**🎯 恭喜！你现在可以运行完整的视频标注流程了！**