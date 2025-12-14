# 🌐 项目网站生成框架

一个简洁、现代、响应式的项目展示网站模板，可快速为任何项目生成专业的单页网站。

## ✨ 特性

- **零依赖** - 纯 HTML/CSS，无需 Node.js、React 或其他框架
- **现代设计** - 渐变背景、卡片布局、悬停动画、完美响应式
- **快速生成** - 修改 JSON 配置，一键生成完整网站
- **高度灵活** - 支持自定义颜色、内容、区块
- **SEO 友好** - 合理的语义化标签和 meta 信息
- **轻量高效** - 单文件 HTML，加载速度极快

## 📁 文件说明

```
templates/
├── project_website_template.html      # HTML 模板（含 CSS 样式）
├── config_example.json                # 配置文件示例
├── generate_website.py                # 自动生成脚本
├── WEBSITE_TEMPLATE_GUIDE.md          # 详细使用指南
└── README_WEBSITE_TEMPLATE.md         # 本文件
```

## 🚀 快速开始（5分钟）

### 1. 创建配置文件

```bash
cp config_example.json my_project.json
```

### 2. 编辑配置

打开 `my_project.json`，修改以下关键内容：

```json
{
  "project_title": "你的项目名称",
  "header": {
    "icon": "🚀",
    "title": "项目标题",
    "tagline": "一句话说明项目核心价值"
  },
  "colors": {
    "primary": "#2563eb",      // 主色调
    "secondary": "#10b981"     // 辅助色
  }
  // ... 更多配置见示例文件
}
```

### 3. 生成网站

```bash
python3 generate_website.py my_project.json my_website.html
```

### 4. 预览效果

```bash
# macOS
open my_website.html

# Linux
xdg-open my_website.html

# Windows
start my_website.html
```

## 🎨 设计演示

原网站效果：https://yizhengyuan.github.io/video-autolabeling-pipeline/

**布局结构：**
```
┌─────────────────────────────────┐
│  Header（渐变背景 + CTA按钮）    │
├─────────────────────────────────┤
│  Features（核心功能卡片）        │
├─────────────────────────────────┤
│  Demo（快速开始步骤）            │
├─────────────────────────────────┤
│  Metrics（数据指标展示）         │
├─────────────────────────────────┤
│  Use Cases（应用场景）           │
├─────────────────────────────────┤
│  CTA（行动号召）                 │
├─────────────────────────────────┤
│  Footer（页脚链接）              │
└─────────────────────────────────┘
```

## 📝 配置文件核心字段

### 颜色主题
```json
"colors": {
  "primary": "#2563eb",       // 主色调（按钮、标题等）
  "primary_dark": "#1e40af",  // 深色变体（渐变、悬停等）
  "secondary": "#10b981"      // 辅助色（徽章、强调等）
}
```

**推荐配色：**
- 科技感：`#2563eb` + `#10b981` （蓝+绿）
- 活力感：`#ea580c` + `#9333ea` （橙+紫）
- 专业感：`#1e40af` + `#0891b2` （深蓝+青）

### Header（顶部）
```json
"header": {
  "icon": "🤖",              // Emoji 图标
  "title": "项目名称",
  "subtitle": "副标题",
  "tagline": "一句话价值描述（20-30字）",
  "buttons": [...]           // 主要 CTA 按钮
}
```

### Features（核心功能）
```json
"features": {
  "title": "核心功能",
  "cards": [
    {
      "icon": "🚀",          // 使用 Emoji
      "title": "功能名称",
      "description": "功能描述（50字内）"
    }
    // 建议 3-6 个功能
  ]
}
```

### Demo（快速开始）
```json
"demo": {
  "title": "快速开始",
  "steps": [
    {
      "title": "Step 1: 安装",
      "code": "pip install your-package",
      "type": "bash"         // bash, python, javascript 等
    }
  ]
}
```

### Metrics（数据展示）
```json
"metrics": {
  "cards": [
    {
      "title": "指标名称",
      "value": "85%",        // 大数字显示
      "description": "说明文字",
      "badge": {
        "text": "推荐",
        "type": "primary"
      }
    }
  ]
}
```

## 💡 使用技巧

### 内容建议
- **标题精炼**：3-5 个字最佳
- **描述简洁**：每个卡片 50 字以内
- **功能数量**：3-6 个核心功能
- **代码简短**：每步不超过 3 行代码
- **场景明确**：3-6 个典型应用场景

### Emoji 推荐
- 技术：⚡ 🔧 🛠️ 💻 🖥️ ⚙️ 🔬
- 速度：🚀 ⚡ 💨 🏃
- AI：🤖 🧠 🎯 🔮
- 数据：📊 📈 💾 🗄️
- 媒体：📹 🖼️ 🎥 📷

### 视觉优化
1. 保持颜色一致性
2. 使用 Emoji 代替图标（简洁、跨平台）
3. 按钮文字用动词开头（"开始使用"、"查看文档"）
4. 文字对比度足够（深色文字 + 浅色背景）

## 🔧 高级定制

### 添加自定义 CSS
在生成的 HTML 文件的 `</style>` 标签前添加：

```css
/* 自定义样式 */
.my-custom-class {
    /* ... */
}
```

### 添加 JavaScript 交互
在生成的 HTML 文件的 `</body>` 标签前添加：

```html
<script>
  // 自定义脚本
  console.log('Hello!');
</script>
```

### 添加 Google Analytics
在 `<head>` 部分添加 GA 跟踪代码。

### 集成第三方服务
- **表单**：Google Forms、Typeform
- **聊天**：Intercom、Crisp
- **统计**：Google Analytics、Plausible

## 📦 部署到 GitHub Pages

### 方法一：直接部署
```bash
# 1. 生成网站
python3 generate_website.py config.json index.html

# 2. 移动到 docs 目录
mkdir -p docs
mv index.html docs/

# 3. 提交到 GitHub
git add docs/index.html
git commit -m "Add project website"
git push

# 4. 在 GitHub 仓库设置中启用 Pages
#    Settings > Pages > Source: docs 目录
```

### 方法二：使用 GitHub Actions

创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy Website
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Generate website
        run: |
          cd templates
          python3 generate_website.py config.json ../docs/index.html
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

## ❓ 常见问题

**Q: 如何更改网站语言？**  
A: 直接在配置文件中使用目标语言编写内容。如需修改 HTML lang 属性，编辑模板第 2 行。

**Q: 可以用于商业项目吗？**  
A: 可以！本模板开源免费，个人和商业项目均可使用。

**Q: 支持多页面吗？**  
A: 当前是单页模板。如需多页面，为每个页面创建独立配置文件。

**Q: 如何添加图片？**  
A: 在配置中使用 HTML `<img>` 标签，或在生成后的 HTML 中手动添加。

**Q: 是否支持深色模式？**  
A: 当前版本不支持。可以手动添加 CSS media query 实现。

## 📚 完整文档

查看 [WEBSITE_TEMPLATE_GUIDE.md](./WEBSITE_TEMPLATE_GUIDE.md) 获取：
- 详细的配置字段说明
- 所有可用选项
- 自定义扩展方法
- 最佳实践建议

## 🤝 示例项目

本框架基于以下项目提炼：
- **MLLM Auto-Labeling** - 原始网站（`config_example.json` 即为此项目配置）
- 查看 `docs/index.html` 查看效果

## 📄 许可证

MIT License - 可自由用于个人和商业项目

---

**Happy Building! 🎉**

如有问题，请参考 `WEBSITE_TEMPLATE_GUIDE.md` 或查看示例配置文件。











