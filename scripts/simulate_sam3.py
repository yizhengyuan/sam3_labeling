#!/usr/bin/env python3
"""
Simulate SAM 3 Output
Generates a dummy JSON with polygon masks to demonstrate the visualization.
"""

import json
import os
import cv2
import numpy as np

def create_mock_polygon(center, radius, num_points=8):
    """Create a simple polygon (octagon)"""
    cx, cy = center
    points = []
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        points.append([x, y])
    return points

def main():
    image_path = "data/WechatIMG6596.jpg"
    output_json = "SAM3_output/mock_sam3_results.json"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    print(f"Simulating SAM 3 results for {image_path} ({w}x{h})...")
    
    # Create some mock objects
    objects = [
        {
            "category": "汽车",
            "center": (w * 0.3, h * 0.6),
            "radius": w * 0.1
        },
        {
            "category": "行人",
            "center": (w * 0.6, h * 0.6),
            "radius": w * 0.05
        },
        {
            "category": "交通标志",
            "center": (w * 0.8, h * 0.4),
            "radius": w * 0.03
        }
    ]
    
    results = []
    for obj in objects:
        # Create polygon points (normalized 0-100)
        poly_pixels = create_mock_polygon(obj["center"], obj["radius"])
        poly_norm = [[p[0]/w*100, p[1]/h*100] for p in poly_pixels]
        
        results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "polygonlabels",
            "value": {
                "points": poly_norm,
                "polygonlabels": [obj["category"]]
            }
        })
        
    data = [{
        "data": {"image": f"/data/local-files/?d={os.path.basename(image_path)}"},
        "predictions": [{"result": results}]
    }]
    
    # Ensure directory exists
    os.makedirs("SAM3_output", exist_ok=True)
    
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)
        
    print(f"✓ Generated mock results: {output_json}")
    
    # Run visualization
    print("Running visualization...")
    # os.system(f"python3 scripts/visualize_result.py")
    
    # We need to manually call the function in visualize_result because main() is hardcoded
    # Let's just import it here
    sys.path.append("scripts")
    from visualize_result import visualize_annotations
    
    output_vis = "SAM3_output/mock_sam3_vis.jpg"
    visualize_annotations(image_path, output_json, output_vis)
    
    print(f"✓ Visualization saved: {output_vis}")

import sys
if __name__ == "__main__":
    main()
