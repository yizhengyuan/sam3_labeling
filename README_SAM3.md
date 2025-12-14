# 🎯 SAM3 Video Auto-Labeling Pipeline

基于 SAM3 的高质量视频自动标注系统，支持目标检测、追踪和 Label Studio 格式输出。

## 📥 输入 / 📤 输出

### 输入
- **视频文件**: MP4, AVI, MOV 等格式
- **目标类别**: 文本提示 (如 "car", "person", "traffic sign")

### 输出
- **JSON**: Label Studio 兼容的标注文件
- **MP4**: 带边界框的可视化标注视频

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision opencv-python numpy filterpy scipy
```

### 2. 运行完整流程 (推荐)
```bash
# 使用现有 SAM3 检测数据 + SORT 追踪器优化
python3 scripts/retrack_with_sort.py SAM3_output/clip_000_every_frame.json \
  --output SAM3_output/my_results.json \
  --video data/D1_video_clips/D1_rand11-15_clip_000.mp4
```

### 3. 查看结果
```bash
# 生成的文件
ls SAM3_output/my_results.*

# 播放标注视频
open SAM3_output/my_results_annotated.mp4
```

## ⚠️ 当前限制

### GPU 依赖问题
由于 SAM3 的 GPU 加速依赖在 Apple Silicon + Python 3.13 环境下存在兼容性问题：

- **triton**: Apple Silicon 不支持
- **decord**: Apple Silicon 不支持
- **numpy版本**: 需要 1.26，但系统有更新版本

### 解决方案
✅ **使用现有数据**: 项目已包含高质量的预检测数据
✅ **SORT追踪器**: 无需 GPU，纯 CPU/MPS 运行
✅ **完整流程**: 从视频到标注结果的端到端解决方案

## 📜 脚本使用指南

### 核心脚本

#### 1. 模拟测试
```bash
# 测试系统功能
python3 scripts/simulate_sam3.py
```

#### 2. SORT 追踪器 (主要工具)
```bash
# 基础用法
python3 scripts/retrack_with_sort.py input.json \
  --output output.json \
  --video your_video.mp4

# 参数调优
python3 scripts/retrack_with_sort.py input.json \
  --output output.json \
  --video video.mp4 \
  --max-age 20 \
  --min-hits 2 \
  --iou-threshold 0.25 \
  --debug
```

#### 3. 结果可视化
```bash
# 生成标注视频
python3 scripts/visualize_sam3_result.py video.mp4 results.json \
  --output annotated_video.mp4
```

### SAM3 原始脚本 (需解决依赖)

#### 视频检测
```bash
# 逐帧检测 (需要 SAM3 依赖)
python3 scripts/sam3_video_tracking.py video.mp4 \
  --text "car" \
  --sample-rate 1 \
  --output detection_results.json
```

#### 图像检测
```bash
# 图像分割
python3 scripts/sam3_auto_labeling.py image.jpg \
  --mode text_to_mask \
  --text_prompt "car . person" \
  --checkpoint checkpoints/sam3/sam3.pt \
  --output masks.json
```

## ⚙️ 参数说明

### SORT 追踪器参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-age` | 30 | 目标消失后保留的帧数 |
| `--min-hits` | 3 | 建立轨迹的最小连续检测数 |
| `--iou-threshold` | 0.3 | IoU 匹配阈值 |
| `--debug` | False | 显示详细调试信息 |

### 检测参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sample-rate` | 5 | 采样率 (1=逐帧, 5=每5帧) |
| `--confidence` | 0.3 | 检测置信度阈值 |
| `--device` | auto | 计算设备 (mps/cuda/cpu) |

## 📊 示例结果

### 处理效果
- **原始检测**: 691 个散乱检测框
- **SORT优化**: 21 条稳定轨迹
- **覆盖时长**: 最长轨迹 285 帧 (99.3% 视频)
- **数据压缩**: JSON 大小减少 13%

### 输出格式
```json
[
  {
    "data": {"video": "/data/local-files/?d=video.mp4"},
    "predictions": [
      {
        "result": [
          {
            "from_name": "box",
            "to_name": "video",
            "type": "videorectangle",
            "value": {
              "sequence": [
                {"frame": 6, "x": 46.2, "y": 32.4, "width": 3.2, "height": 4.9, "time": 0.27}
              ],
              "labels": ["car"]
            },
            "id": "track_0"
          }
        ]
      }
    ]
  }
]
```

## 🔧 故障排除

### 常见问题
1. **SAM3 依赖错误** → 使用现有数据 `SAM3_output/clip_000_every_frame.json`
2. **内存不足** → 增大 `--sample-rate` 参数
3. **设备选择** → Apple Silicon 使用 `--device mps`

### 环境测试
```bash
python3 -c "
import torch, cv2, filterpy
print('✅ 基本环境就绪')
print(f'MPS可用: {torch.backends.mps.is_available()}')
"
```

## 📁 项目结构
```
├── scripts/
│   ├── retrack_with_sort.py      # SORT追踪器 (主要工具)
│   ├── simulate_sam3.py          # 模拟测试
│   ├── sam3_video_tracking.py    # SAM3视频检测
│   └── visualize_sam3_result.py  # 结果可视化
├── SAM3_output/                  # 输出目录
├── data/D1_video_clips/          # 测试视频
└── checkpoints/sam3/            # SAM3模型文件
```

## 📝 使用流程

1. **准备视频**: 将视频放入 `data/D1_video_clips/`
2. **运行追踪**: 使用 `retrack_with_sort.py` 处理
3. **查看结果**: 检查生成的 JSON 和 MP4 文件
4. **导入标注**: 将 JSON 导入 Label Studio

## 🎯 快速验证

```bash
# 运行完整示例
python3 scripts/retrack_with_sort.py SAM3_output/clip_000_every_frame.json \
  --output SAM3_output/demo_results.json \
  --video data/D1_video_clips/D1_rand11-15_clip_000.mp4

# 验证输出
ls SAM3_output/demo_results.*
open SAM3_output/demo_results_annotated.mp4
```

---

**🎉 准备好了！开始你的视频标注之旅吧！**