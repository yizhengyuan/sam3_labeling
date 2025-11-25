#!/bin/bash
# 快速创建项目网站脚本

echo "🌐 项目网站生成器"
echo "=================="
echo ""

# 检查是否提供了项目名称
if [ -z "$1" ]; then
    echo "使用方法: ./quick_create_website.sh <项目名称>"
    echo ""
    echo "示例:"
    echo "  ./quick_create_website.sh my_awesome_project"
    echo ""
    exit 1
fi

PROJECT_NAME=$1
CONFIG_FILE="${PROJECT_NAME}_config.json"
OUTPUT_FILE="${PROJECT_NAME}_website.html"

# 检查配置文件是否已存在
if [ -f "$CONFIG_FILE" ]; then
    echo "⚠️  配置文件 $CONFIG_FILE 已存在"
    echo "正在使用现有配置文件生成网站..."
else
    echo "📄 创建新配置文件: $CONFIG_FILE"
    cp config_simple_example.json "$CONFIG_FILE"
    echo "✅ 已创建配置文件模板"
    echo ""
    echo "📝 请编辑 $CONFIG_FILE 填入您的项目信息"
    echo ""
    echo "编辑完成后，再次运行本脚本生成网站："
    echo "  ./quick_create_website.sh $PROJECT_NAME"
    echo ""
    exit 0
fi

# 生成网站
echo ""
echo "🔧 生成网站..."
python3 generate_website.py "$CONFIG_FILE" "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 完成！"
    echo ""
    echo "生成的文件："
    echo "  - 配置文件: $CONFIG_FILE"
    echo "  - 网站文件: $OUTPUT_FILE"
    echo ""
    echo "下一步："
    echo "  1. 在浏览器中预览: open $OUTPUT_FILE"
    echo "  2. 部署到 GitHub Pages"
    echo "  3. 与世界分享您的项目！"
    echo ""
else
    echo ""
    echo "❌ 生成失败，请检查配置文件格式"
    echo ""
fi










