# 项目网站模板使用指南

这是一个简洁、现代、响应式的项目展示网站模板，可以快速为任何项目生成专业的单页网站。

## 📁 文件说明

- **`project_website_template.html`** - HTML模板文件（包含所有样式和结构）
- **`config_example.json`** - 配置文件示例（包含所有可配置的内容）
- **`generate_website.py`** - 自动生成脚本（将配置转换为完整HTML）
- **`WEBSITE_TEMPLATE_GUIDE.md`** - 本使用指南

## 🚀 快速开始

### 方法一：使用生成脚本（推荐）

1. **复制示例配置文件**
```bash
cp config_example.json my_project.json
```

2. **编辑配置文件**
```bash
# 使用任何文本编辑器编辑 my_project.json
# 修改项目名称、描述、功能等内容
```

3. **生成网站**
```bash
python3 generate_website.py my_project.json my_website.html
```

4. **在浏览器中预览**
```bash
open my_website.html  # macOS
# 或直接拖拽 HTML 文件到浏览器
```

### 方法二：手动编辑模板

如果您偏好手动编辑：

1. 复制 `project_website_template.html` 
2. 搜索并替换所有 `{{...}}` 占位符
3. 手动填入您的项目内容

## 📝 配置文件结构说明

### 1. 基本信息

```json
{
  "meta_description": "网站描述（用于SEO）",
  "meta_keywords": "关键词1, 关键词2, 关键词3",
  "project_title": "网站标题（显示在浏览器标签）"
}
```

### 2. 颜色主题

```json
{
  "colors": {
    "primary": "#2563eb",       // 主色调（蓝色系）
    "primary_dark": "#1e40af",  // 深色主题
    "secondary": "#10b981"      // 辅助色（绿色系）
  }
}
```

**推荐配色方案：**
- 科技感：蓝色 `#2563eb` + 绿色 `#10b981`
- 活力感：橙色 `#ea580c` + 紫色 `#9333ea`
- 专业感：深蓝 `#1e40af` + 青色 `#0891b2`
- 自然感：绿色 `#059669` + 黄色 `#d97706`

### 3. Header（顶部区域）

```json
{
  "header": {
    "icon": "🤖",              // Emoji 图标
    "title": "项目名称",
    "subtitle": "项目副标题",
    "tagline": "一句话介绍项目的核心价值",
    "buttons": [
      {
        "text": "按钮文字",
        "url": "https://...",
        "type": "primary"      // primary 或 secondary
      }
    ]
  }
}
```

### 4. Features（核心功能）

```json
{
  "features": {
    "title": "功能区标题",
    "subtitle": "功能区副标题",
    "cards": [
      {
        "icon": "🚀",          // 使用 Emoji
        "title": "功能名称",
        "description": "功能详细描述"
      }
      // 建议 3-6 个功能卡片
    ]
  }
}
```

**常用 Emoji 推荐：**
- 技术类：⚡ 🔧 🛠️ 🔬 💻 🖥️ ⚙️
- 速度类：🚀 ⚡ 💨 🏃
- AI/机器人：🤖 🧠 🎯 🔮
- 数据类：📊 📈 📉 💾 🗄️
- 媒体类：📹 🖼️ 🎥 📷 🎬

### 5. Demo（快速开始）

```json
{
  "demo": {
    "title": "快速开始标题",
    "subtitle": "快速开始副标题",
    "steps": [
      {
        "title": "Step 1: 安装",
        "code": "pip install your-package",
        "type": "bash"         // bash, python, javascript 等
      }
    ],
    "footer_note": "✨ 可选的底部提示文字"
  }
}
```

### 6. Metrics（数据/模型）

```json
{
  "metrics": {
    "title": "指标区标题",
    "subtitle": "指标区副标题",
    "cards": [
      {
        "title": "指标名称",
        "value": "85%",        // 大数字显示
        "description": "指标说明",
        "badge": {             // 可选徽章
          "text": "推荐",
          "type": "primary"    // primary 或 secondary
        }
      }
    ]
  }
}
```

### 7. Use Cases（应用场景）

```json
{
  "use_cases": {
    "title": "应用场景标题",
    "subtitle": "应用场景副标题",
    "items": [
      {
        "icon": "🚗",
        "title": "场景名称",
        "description": "场景描述"
      }
      // 建议 3-6 个场景
    ]
  }
}
```

### 8. CTA（行动号召）

```json
{
  "cta": {
    "title": "号召性标题",
    "subtitle": "号召性副标题",
    "buttons": [
      {
        "text": "立即开始",
        "url": "https://...",
        "type": "primary"
      }
    ]
  }
}
```

### 9. Footer（页脚）

```json
{
  "footer": {
    "content": "版权信息 | <a href='...'>链接</a>"
  }
}
```

## 🎨 设计特点

- ✅ **响应式设计** - 完美适配手机、平板、桌面
- ✅ **现代化样式** - 渐变背景、卡片阴影、悬停动画
- ✅ **简洁大方** - 清晰的视觉层次，专业的排版
- ✅ **高性能** - 纯HTML/CSS，无依赖，加载迅速
- ✅ **SEO友好** - 合理的语义化标签和meta信息

## 📐 布局结构

网站采用经典的单页布局：

```
┌─────────────────────────────────┐
│  Header（渐变背景 + CTA按钮）    │
├─────────────────────────────────┤
│  Features（功能卡片网格）        │
├─────────────────────────────────┤
│  Demo（代码示例 + 步骤）         │
├─────────────────────────────────┤
│  Metrics（数据/指标展示）        │
├─────────────────────────────────┤
│  Use Cases（应用场景）           │
├─────────────────────────────────┤
│  CTA（再次号召行动）             │
├─────────────────────────────────┤
│  Footer（页脚链接）              │
└─────────────────────────────────┘
```

## 💡 最佳实践

### 内容建议

1. **Header 标题**：简短有力（3-5个字）
2. **Tagline**：一句话说清楚项目价值（20-30字）
3. **功能数量**：3-6个核心功能
4. **代码示例**：每步不超过3行代码
5. **应用场景**：3-6个典型场景

### 视觉建议

1. **颜色一致性**：全站使用相同的主色调
2. **图标统一**：优先使用 Emoji（简洁、跨平台）
3. **文字精炼**：每个卡片描述控制在 50 字以内
4. **按钮文字**：动词开头（"开始使用"、"查看文档"）

### 技术建议

1. **图片优化**：如需添加图片，使用压缩后的格式
2. **外部链接**：GitHub、文档等链接确保有效
3. **浏览器测试**：在 Chrome、Safari、Firefox 中测试
4. **移动端测试**：使用浏览器开发者工具检查响应式

## 🔧 自定义扩展

### 添加新的区块

如需添加新区块（如"团队介绍"、"定价方案"等）：

1. 在 `project_website_template.html` 中复制一个现有 section
2. 修改样式类名和结构
3. 在配置文件中添加对应的数据结构
4. 在 `generate_website.py` 中添加生成逻辑

### 修改颜色方案

除了修改 `colors` 配置外，还可以直接修改 CSS 变量：

```css
:root {
    --primary: #your-color;
    --primary-dark: #your-dark-color;
    --secondary: #your-secondary-color;
}
```

### 更改字体

在 CSS 中修改 `font-family`：

```css
body {
    font-family: 'Your Font', -apple-system, sans-serif;
}
```

## 📦 示例项目

项目已包含完整示例：
- **配置文件**：`config_example.json`（MLLM 自动标注项目）
- **生成结果**：可以直接运行生成脚本查看效果

## ❓ 常见问题

**Q: 如何更改网站语言？**  
A: 在配置文件中直接使用目标语言编写内容即可。HTML 的 `lang` 属性可在模板中修改。

**Q: 可以添加 JavaScript 交互吗？**  
A: 可以！在模板底部 `</body>` 前添加 `<script>` 标签。

**Q: 如何部署到 GitHub Pages？**  
A: 将生成的 HTML 重命名为 `index.html`，放入仓库的 `docs/` 目录，在 GitHub 仓库设置中启用 Pages。

**Q: 支持多页面吗？**  
A: 当前是单页模板。如需多页面，建议为每个页面创建独立配置文件。

**Q: 可以集成 Google Analytics 吗？**  
A: 可以！在模板 `<head>` 部分添加 GA 跟踪代码即可。

## 🤝 贡献

欢迎提交改进建议：
- 新的配色方案
- 新的区块类型
- 性能优化
- Bug 修复

## 📄 许可证

本模板基于原项目许可证发布，可自由用于个人和商业项目。

---

**Happy Building! 🎉**

如有问题，请查看示例配置文件或阅读生成脚本的注释。










