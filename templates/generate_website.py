#!/usr/bin/env python3
"""
项目网站生成器
从JSON配置文件生成完整的项目展示网站

使用方法:
    python3 generate_website.py config.json output.html
    python3 generate_website.py config.json  # 默认输出到 index.html
"""

import json
import sys
import os
from pathlib import Path


def load_config(config_file):
    """加载JSON配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_template(template_file):
    """加载HTML模板"""
    with open(template_file, 'r', encoding='utf-8') as f:
        return f.read()


def generate_buttons_html(buttons):
    """生成按钮HTML"""
    html = ""
    for btn in buttons:
        btn_class = f"btn btn-{btn['type']}"
        html += f'                <a href="{btn["url"]}" class="btn {btn_class}">\n'
        html += f'                    {btn["text"]}\n'
        html += f'                </a>\n'
    return html.strip()


def generate_feature_cards_html(cards):
    """生成功能卡片HTML"""
    html = ""
    for card in cards:
        html += '                <div class="feature-card">\n'
        html += f'                    <div class="icon">{card["icon"]}</div>\n'
        html += f'                    <h3>{card["title"]}</h3>\n'
        html += f'                    <p>\n'
        html += f'                        {card["description"]}\n'
        html += f'                    </p>\n'
        html += '                </div>\n\n'
    return html.strip()


def generate_demo_steps_html(steps, footer_note=None):
    """生成演示步骤HTML"""
    html = ""
    for i, step in enumerate(steps):
        margin_style = "" if i == 0 else " style=\"margin: 30px 0 20px 0;\""
        html += f'                <h3{margin_style}>{step["title"]}</h3>\n'
        html += '                <div class="code-block">\n'
        html += f'{step["code"]}\n'
        html += '                </div>\n\n'
    
    if footer_note:
        html += f'                <p style="margin-top: 30px; text-align: center; font-size: 1.1rem;">\n'
        html += f'                    {footer_note}\n'
        html += '                </p>\n'
    
    return html.strip()


def generate_metric_cards_html(cards):
    """生成指标卡片HTML"""
    html = ""
    for card in cards:
        html += '                <div class="metric-card">\n'
        html += f'                    <h3>{card["title"]}</h3>\n'
        html += f'                    <div class="value">{card["value"]}</div>\n'
        html += f'                    <p class="description">\n'
        html += f'                        {card["description"]}\n'
        html += '                    </p>\n'
        
        if "badge" in card and card["badge"]:
            badge_class = f'badge-{card["badge"]["type"]}'
            html += f'                    <span class="badge {badge_class}">{card["badge"]["text"]}</span>\n'
        
        html += '                </div>\n\n'
    return html.strip()


def generate_use_cases_html(items):
    """生成使用场景HTML"""
    html = ""
    for item in items:
        html += '                <div class="use-case">\n'
        html += f'                    <h3>{item["icon"]} {item["title"]}</h3>\n'
        html += f'                    <p>{item["description"]}</p>\n'
        html += '                </div>\n\n'
    return html.strip()


def generate_website(config, template):
    """根据配置和模板生成网站HTML"""
    html = template
    
    # 替换基础信息
    html = html.replace('{{meta_description}}', config['meta_description'])
    html = html.replace('{{meta_keywords}}', config['meta_keywords'])
    html = html.replace('{{project_title}}', config['project_title'])
    
    # 替换颜色
    html = html.replace('{{color_primary}}', config['colors']['primary'])
    html = html.replace('{{color_primary_dark}}', config['colors']['primary_dark'])
    html = html.replace('{{color_secondary}}', config['colors']['secondary'])
    
    # 替换 Header
    html = html.replace('{{header_icon}}', config['header']['icon'])
    html = html.replace('{{header_title}}', config['header']['title'])
    html = html.replace('{{header_subtitle}}', config['header']['subtitle'])
    html = html.replace('{{header_tagline}}', config['header']['tagline'])
    html = html.replace('{{header_buttons}}', generate_buttons_html(config['header']['buttons']))
    
    # 替换 Features
    html = html.replace('{{features_title}}', config['features']['title'])
    html = html.replace('{{features_subtitle}}', config['features']['subtitle'])
    html = html.replace('{{features_cards}}', generate_feature_cards_html(config['features']['cards']))
    
    # 替换 Demo
    html = html.replace('{{demo_title}}', config['demo']['title'])
    html = html.replace('{{demo_subtitle}}', config['demo']['subtitle'])
    footer_note = config['demo'].get('footer_note')
    html = html.replace('{{demo_steps}}', generate_demo_steps_html(config['demo']['steps'], footer_note))
    
    # 替换 Metrics
    html = html.replace('{{metrics_title}}', config['metrics']['title'])
    html = html.replace('{{metrics_subtitle}}', config['metrics']['subtitle'])
    html = html.replace('{{metrics_cards}}', generate_metric_cards_html(config['metrics']['cards']))
    
    # 替换 Use Cases
    html = html.replace('{{usecases_title}}', config['use_cases']['title'])
    html = html.replace('{{usecases_subtitle}}', config['use_cases']['subtitle'])
    html = html.replace('{{usecases_items}}', generate_use_cases_html(config['use_cases']['items']))
    
    # 替换 CTA
    html = html.replace('{{cta_title}}', config['cta']['title'])
    html = html.replace('{{cta_subtitle}}', config['cta']['subtitle'])
    html = html.replace('{{cta_buttons}}', generate_buttons_html(config['cta']['buttons']))
    
    # 替换 Footer
    html = html.replace('{{footer_content}}', config['footer']['content'])
    
    return html


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python3 generate_website.py config.json [output.html]")
        print("\n示例:")
        print("  python3 generate_website.py config_example.json")
        print("  python3 generate_website.py my_project.json my_project.html")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "index.html"
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    template_file = script_dir / "project_website_template.html"
    
    if not os.path.exists(config_file):
        print(f"❌ 错误: 配置文件 '{config_file}' 不存在")
        sys.exit(1)
    
    if not os.path.exists(template_file):
        print(f"❌ 错误: 模板文件 '{template_file}' 不存在")
        sys.exit(1)
    
    print(f"📖 读取配置文件: {config_file}")
    config = load_config(config_file)
    
    print(f"📄 读取模板文件: {template_file}")
    template = load_template(template_file)
    
    print("🔧 生成网站...")
    html = generate_website(config, template)
    
    print(f"💾 保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ 成功! 网站已生成到 {output_file}")
    print(f"🌐 在浏览器中打开: file://{os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()











