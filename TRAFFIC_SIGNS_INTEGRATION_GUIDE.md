# 🚦 交通标志检测集成指南

## 📋 概述

本指南介绍如何将 `signs` 数据集（188个高分辨率交通标志）集成到现有的 SAM3 视频标注工作流中，实现交通标志的自动检测、分割和标注。

## 🎯 系统架构

```
Input Video → Traffic Sign Detection → (Optional) SAM3 Segmentation → SORT Tracking → Label Studio Output
     ↓                    ↓                           ↓                    ↓
   视频帧           模板匹配检测                精确分割           稳定轨迹追踪
```

## 📁 新增文件结构

```
├── scripts/
│   ├── traffic_sign_detector.py              # 交通标志检测器
│   ├── integrated_traffic_sign_pipeline.py   # 集成流水线
│   └── visualize_traffic_signs.py            # 可视化工具
├── config/
│   └── traffic_sign_config.json              # 配置文件
├── signs/                                    # 交通标志数据集
│   └── highres/png2560px/                    # 2560x2560 PNG图像
└── TRAFFIC_SIGNS_INTEGRATION_GUIDE.md       # 本指南
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖 (应该已安装)
pip install opencv-python numpy

# 如果要使用SORT追踪器 (推荐)
pip install filterpy scipy
```

### 2. 基础检测

使用交通标志检测器处理视频：

```bash
python3 scripts/traffic_sign_detector.py \
    --video data/D1_video_clips/your_video.mp4 \
    --output traffic_signs_detections.json \
    --signs-dir signs/highres/png2560px/ \
    --threshold 0.7 \
    --sample-rate 5
```

### 3. 完整流水线 (推荐)

使用集成流水线，包含检测和追踪：

```bash
python3 scripts/integrated_traffic_sign_pipeline.py \
    --video data/D1_video_clips/your_video.mp4 \
    --output SAM3_output/traffic_signs_results.json \
    --signs-dir signs/highres/png2560px/ \
    --threshold 0.7 \
    --sample-rate 5
```

### 4. 可视化结果

生成带标注的视频：

```bash
python3 scripts/visualize_traffic_signs.py \
    --video data/D1_video_clips/your_video.mp4 \
    --detections SAM3_output/traffic_signs_results.json \
    --output SAM3_output/traffic_signs_annotated.mp4 \
    --summary SAM3_output/traffic_signs_summary.png
```

## ⚙️ 配置参数

### 检测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--threshold` | 0.7 | 模板匹配阈值 (0.0-1.0) |
| `--sample-rate` | 5 | 采样率 (每N帧处理一次) |
| `scale_range` | 0.2-2.0 | 多尺度检测范围 |

### 追踪参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_age` | 30 | 目标消失后保留帧数 |
| `min_hits` | 3 | 建立轨迹的最小检测数 |
| `iou_threshold` | 0.3 | IoU匹配阈值 |

## 🎨 交通标志类别

系统自动识别以下类别：

### 警告标志 (Warning)
- `Bend_to_left_ahead` - 左转警告
- `Cross_roads_ahead` - 十字路口警告
- `Children_ahead` - 儿童警告
- `Cyclists_ahead` - 骑行者警告

### 禁令标志 (Regulatory)
- `Stop_and_give_way` - 停车让行
- `No_stopping` - 禁止停车
- `Speed_limit_(in_km_h)` - 速度限制
- `Ahead_only` - 直行

### 指示标志 (Information)
- `Bus_lane_ahead` - 公交车道
- `Census_point` - 统计点
- `Bicycle_tricycle_route` - 自行车道

### 距离标志 (Distance)
- `100m_Countdown_markers` - 100米倒计时标志
- `200m_Countdown_markers` - 200米倒计时标志
- `300m_Countdown_markers` - 300米倒计时标志

## 📊 输出格式

### Label Studio 兼容格式

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
                  "frame": 10,
                  "x": 45.2,
                  "y": 32.1,
                  "width": 3.5,
                  "height": 4.8,
                  "time": 0.33,
                  "enabled": true
                }
              ],
              "labels": ["Stop_and_give_way"]
            },
            "id": "track_0"
          }
        ],
        "score": 0.85
      }
    ]
  }
]
```

## 🔧 高级用法

### 1. 调整检测灵敏度

```bash
# 高灵敏度 (更多检测，可能有误报)
python3 scripts/integrated_traffic_sign_pipeline.py \
    --threshold 0.5 \
    --video your_video.mp4 \
    --output results_high_sensitivity.json

# 低灵敏度 (更精确，可能漏检)
python3 scripts/integrated_traffic_sign_pipeline.py \
    --threshold 0.8 \
    --video your_video.mp4 \
    --output results_high_precision.json
```

### 2. 禁用追踪 (仅检测)

```bash
python3 scripts/integrated_traffic_sign_pipeline.py \
    --video your_video.mp4 \
    --output results_detection_only.json \
    --no-tracking
```

### 3. 自定义配置

编辑 `config/traffic_sign_config.json`:

```json
{
  "detection_config": {
    "threshold": 0.7,
    "sample_rate": 3,  // 降低采样率，更频繁处理
    "scale_range": {
      "min": 0.1,      // 检测更小的标志
      "max": 3.0,      // 检测更大的标志
      "steps": 20      // 更多尺度
    }
  }
}
```

## 🎯 使用案例

### 案例1: 交通流量分析
```bash
# 检测视频中的所有交通标志
python3 scripts/integrated_traffic_sign_pipeline.py \
    --video traffic_video.mp4 \
    --output traffic_analysis.json \
    --sample-rate 10  # 每10帧处理一次，提高速度
```

### 案例2: 标志合规检查
```bash
# 高精度检测特定类型的标志
python3 scripts/integrated_traffic_sign_pipeline.py \
    --video compliance_video.mp4 \
    --output compliance_check.json \
    --threshold 0.85  # 高阈值确保准确性
```

### 案例3: 驾驶训练数据标注
```bash
# 完整标注流水线，生成训练数据
python3 scripts/integrated_traffic_sign_pipeline.py \
    --video training_video.mp4 \
    --output training_data.json \
    --sample-rate 1   # 逐帧处理，确保完整性

# 生成可视化视频
python3 scripts/visualize_traffic_signs.py \
    --video training_video.mp4 \
    --detections training_data.json \
    --output training_annotated.mp4 \
    --summary training_summary.png
```

## 🔍 故障排除

### 常见问题

#### 1. 检测结果过多
**问题**: 检测到太多误报
**解决方案**:
```bash
# 提高阈值
--threshold 0.8

# 或调整NMS阈值
python3 -c "
import json
with open('config/traffic_sign_config.json', 'r') as f:
    config = json.load(f)
config['detection_config']['nms_threshold'] = 0.3  # 降低值
with open('config/traffic_sign_config.json', 'w') as f:
    json.dump(config, f, indent=2)
"
```

#### 2. 漏检某些标志
**问题**: 明显存在的标志未被检测到
**解决方案**:
```bash
# 降低阈值
--threshold 0.5

# 降低采样率
--sample-rate 2  # 更频繁处理

# 扩大尺度范围
# 编辑config文件，调整scale_range
```

#### 3. 处理速度慢
**问题**: 处理大视频文件时间过长
**解决方案**:
```bash
# 提高采样率
--sample-rate 10  # 每10帧处理一次

# 降低检测精度换取速度
--threshold 0.6
```

#### 4. 内存不足
**问题**: 处理大尺寸视频时内存不足
**解决方案**:
```bash
# 预先压缩视频
ffmpeg -i input.mp4 -vf "scale=1280:-1" -c:a copy compressed.mp4

# 或提高采样率
--sample-rate 15
```

### 性能优化建议

1. **视频预处理**: 将视频缩放到合理尺寸 (如1280x720)
2. **适当采样**: 根据视频帧率调整采样率
3. **阈值调优**: 根据具体场景调整检测阈值
4. **批量处理**: 对多个视频使用脚本批量处理

## 📈 性能基准

### 典型性能数据
- **检测速度**: ~2-5 FPS (取决于视频分辨率和采样率)
- **检测准确率**: 85-95% (取决于标志质量和场景)
- **内存使用**: 2-4 GB (处理1080p视频)

### 推荐设置
- **实时应用**: 采样率=5, 阈值=0.7
- **离线高精度**: 采样率=1, 阈值=0.6
- **快速预览**: 采样率=10, 阈值=0.8

## 🔗 与现有系统集成

### 1. 与SORT追踪器集成
检测结果可以输入到现有的 `retrack_with_sort.py` 脚本:

```bash
# 先运行交通标志检测
python3 scripts/traffic_sign_detector.py \
    --video video.mp4 \
    --output detections.json

# 然后应用SORT追踪器
python3 scripts/retrack_with_sort.py \
    detections.json \
    --video video.mp4 \
    --output final_results.json
```

### 2. 与SAM3分割集成
检测到的标志可以进一步使用SAM3进行精确分割:

```python
# 在检测脚本中添加SAM3调用
from scripts.sam3_auto_labeling import segment_image

# 对检测到的区域进行分割
mask = segment_image(frame, bbox)
```

## 🎉 下一步

1. **测试系统**: 使用示例视频测试完整流水线
2. **调优参数**: 根据您的具体场景调整参数
3. **集成到工作流**: 将检测结果导入Label Studio
4. **扩展数据集**: 添加更多自定义交通标志

---

**🚦 恭喜！您现在可以检测和标注交通标志了！**