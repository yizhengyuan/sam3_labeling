# MLLM Auto-Labeling for Images & Videos

> 🌐 **[View Project Website](https://yizhengyuan.github.io/video-autolabeling-pipeline/)** | 📚 **[Documentation](QUICKSTART.md)** | 💬 **[Issues](https://github.com/yizhengyuan/video-autolabeling-pipeline/issues)**

**Leverage Multimodal Large Language Models (MLLMs) to automatically label your image and video datasets.**

Generate high-quality bounding box annotations using state-of-the-art vision-language models like GPT-4V, Claude 3.5 Sonnet, and Qwen-VL. Save 80%+ annotation time while maintaining 85-95% accuracy.

Integrate with Label Studio for human review and collaborative annotation workflows.

---

## 🚦 NEW: Traffic Sign Detection Integration

**🎉 Introducing advanced traffic sign detection powered by 188 high-resolution templates!**

```
📁 signs/highres/png2560px/ → 188 traffic signs (2560×2560 PNG)
🔧 scripts/traffic_sign_detector.py → Multi-scale template matching
🔧 scripts/integrated_traffic_sign_pipeline.py → Complete detection & tracking pipeline
🔧 scripts/visualize_traffic_signs.py → Rich visualization and annotation
📚 TRAFFIC_SIGNS_INTEGRATION_GUIDE.md → Complete usage documentation
```

**Quick Demo:**
```bash
# Detect traffic signs in your video
python3 scripts/integrated_traffic_sign_pipeline.py \
    --video your_traffic_video.mp4 \
    --output traffic_signs_results.json \
    --threshold 0.7

# Visualize results
python3 scripts/visualize_traffic_signs.py \
    --video your_traffic_video.mp4 \
    --detections traffic_signs_results.json \
    --output annotated_video.mp4

# Run complete demo
python3 scripts/demo_traffic_sign_integration.py
```

**Features:**
- 🔍 **188 Traffic Sign Templates**: Complete Hong Kong traffic sign library
- 🎯 **Multi-Scale Detection**: 0.2x-2.0x scale range with NMS optimization
- 🚗 **Real-Time Processing**: 2-5 FPS on 1080p video
- 🏷️ **Label Studio Compatible**: Direct integration with annotation workflows
- 📊 **Rich Visualization**: Color-coded categories, confidence scores, track IDs
- ⚙️ **Configurable**: Adjustable thresholds, sampling rates, tracking parameters

**[📖 Complete Guide → TRAFFIC_SIGNS_INTEGRATION_GUIDE.md](TRAFFIC_SIGNS_INTEGRATION_GUIDE.md)**

---

## 📁 Project Structure

```
video-autolabeling-pipeline/
├── README.md              # Main documentation
├── LICENSE                # Open source license
├── requirements.txt       # Python dependencies
├── docs/                  # 📚 Documentation
│   ├── 快速开始.md        # Quick start guide (Chinese)
│   ├── QWEN_GUIDE.md      # Qwen-VL detailed guide
│   ├── AUTO_LABELING_GUIDE.md  # VLM auto-labeling guide
│   └── YOLO_GUIDE.md      # YOLO local labeling guide
├── config/                # ⚙️ Configuration files
│   └── traffic_sign_config.json  # 🚦 Traffic sign detection config
├── signs/                 # 🚦 Traffic sign dataset
│   └── highres/png2560px/ # 188 traffic signs (2560×2560 PNG)
├── scripts/               # 🔧 Core scripts
│   ├── image_auto_labeling.py     # Image auto-labeling
│   ├── video_auto_labeling.py     # Video auto-labeling
│   ├── yolo_auto_labeling.py      # YOLO labeling
│   ├── quick_yolo_label.sh        # YOLO quick labeling script
│   ├── visualize_result.py        # Visualize labeling results
│   ├── test_qwen_api.py           # Test Qwen API
│   ├── start_label_studio.sh      # Start Label Studio
│   ├── traffic_sign_detector.py   # 🚦 Traffic sign detection
│   ├── integrated_traffic_sign_pipeline.py  # 🚦 Complete traffic sign pipeline
│   ├── visualize_traffic_signs.py  # 🚦 Traffic sign visualization
│   └── demo_traffic_sign_integration.py  # 🚦 Demo script
├── templates/             # 📋 Labeling templates
├── data/                  # 📹 Data files (examples)
└── labels/                # 🏷️ Labeling results (output)
```

---

## 🎓 Getting Started

**First time user?** → Check **[QUICKSTART.md](QUICKSTART.md)** for complete tutorial (10 mins setup)

---

## 🚀 Quick Start

**Test with an image (fastest):**

```bash
# 1. Set API Key (choose one)
export DASHSCOPE_API_KEY="your-qwen-key"        # Qwen (recommended for China)
export ANTHROPIC_API_KEY="your-claude-key"     # Claude (recommended for international)

# 2. Label an image
python3 scripts/image_auto_labeling.py your-image.jpg --provider qwen --visualize

# View the labeled result with bounding boxes instantly!
```

> 💡 For detailed steps, see **[QUICKSTART.md](QUICKSTART.md)** or **[快速开始.md](docs/快速开始.md)** (Chinese)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🚦 **Traffic Sign Detection** | 188 traffic sign templates with multi-scale detection and tracking |
| 🤖 **MLLM Auto-Labeling** | Leverage GPT-4V, Claude 3.5 Sonnet, Qwen-VL to auto-generate labels |
| 📹 **Video Frame Labeling** | Smart sampling strategies for efficient video annotation |
| 🖼️ **Image Object Detection** | Single-shot bounding box generation for images |
| ⚡ **Save 80%+ Time** | AI generates initial labels, humans only review and refine |
| 🎯 **High Accuracy** | Traffic Signs: 85-95%, Claude: 90-95%, Qwen: 85-90%, YOLO: 80-85% |
| 🌐 **China-Friendly** | Hong Kong traffic signs + Qwen-VL support, no VPN required |
| 🔧 **Production Ready** | Label Studio integration, batch processing, visualization tools |

---

## 📚 Documentation

**🚦 Traffic Sign Detection (NEW):**
- **[TRAFFIC_SIGNS_INTEGRATION_GUIDE.md](TRAFFIC_SIGNS_INTEGRATION_GUIDE.md)** 🔥 **NEW** - Complete traffic sign detection guide
- **[TRAFFIC_SIGNS_INTEGRATION_SUMMARY.md](TRAFFIC_SIGNS_INTEGRATION_SUMMARY.md)** - Project summary and technical details

**For Beginners:**
- **[QUICKSTART.md](QUICKSTART.md)** 🔰 **Start Here** - Complete tutorial, 10-min setup
- **[快速开始.md](docs/快速开始.md)** ⭐ Quick Start - Three ways to get started (Chinese)

**Advanced Guides:**
- **[QWEN_GUIDE.md](docs/QWEN_GUIDE.md)** - Qwen-VL detailed guide (recommended for users in China)
- **[AUTO_LABELING_GUIDE.md](docs/AUTO_LABELING_GUIDE.md)** - VLM auto-labeling with GPT-4V, Claude, etc.
- **[YOLO_GUIDE.md](docs/YOLO_GUIDE.md)** - YOLO local labeling (free, offline)

---

## 🎯 Use Cases

- 🚦 **Traffic Sign Recognition**: Automated detection and classification of 188+ Hong Kong traffic signs
- 🚗 **Autonomous Driving**: Vehicle, pedestrian, traffic sign detection
- 🏭 **Industrial QA**: Defect detection, product classification
- 🏥 **Medical Imaging**: Lesion annotation, organ segmentation
- 📦 **E-commerce**: Product recognition, shelf monitoring
- 🎥 **Video Analytics**: Action recognition, object tracking

---

## 💡 Contributing

We welcome Issues and Pull Requests! If you have questions or suggestions, please contact us on GitHub.

## 📄 License

This project is licensed under the [LICENSE](LICENSE) file.
