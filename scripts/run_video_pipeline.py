#!/usr/bin/env python3
"""
One-Click Video Labeling Pipeline
Runs the auto-labeling (MLLM/SAM3) and generates a visualized video in one step.
"""

import os
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="One-Click Video Labeling Pipeline")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--provider", default="qwen", choices=["openai", "anthropic", "qwen", "sam3", "mock"], help="Labeling provider")
    parser.add_argument("--sample-rate", type=int, default=5, help="Frame sampling rate")
    parser.add_argument("--output_dir", default="output_results", help="Directory to save results")
    
    # SAM3 specific
    parser.add_argument("--sam3_checkpoint", help="Path to SAM3 checkpoint (required if provider=sam3)")
    parser.add_argument("--text_prompt", help="Text prompt for SAM3")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu", "mps"], help="Device to use (cuda, cpu, mps)")
    
    args = parser.parse_args()
    
    # Setup paths
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    basename = video_path.stem
    json_output = output_dir / f"{basename}_labels.json"
    video_output = output_dir / f"{basename}_annotated.mp4"
    
    print("="*60)
    print("🚀 Starting Video Labeling Pipeline")
    print(f"Input: {video_path}")
    print(f"Provider: {args.provider}")
    print("="*60)
    
    # 1. Run Labeling
    print("\n[Step 1/2] Generating Labels...")
    
    if args.provider == "sam3":
        # Use SAM3 script
        if not args.sam3_checkpoint:
            print("❌ Error: --sam3_checkpoint is required for SAM3 provider")
            return
            
        if not args.text_prompt:
            print("❌ Error: --text_prompt is required for SAM3 video tracking")
            return
            
        cmd = f"{sys.executable} scripts/sam3_video_tracking.py {video_path} --text '{args.text_prompt}' --checkpoint {args.sam3_checkpoint} --output {json_output} --sample-rate {args.sample_rate}"
        if args.device:
            cmd += f" --device {args.device}"
            
    elif args.provider == "mock":
        # Generate mock data for demonstration
        print("⚠️  Using MOCK provider for demonstration...")
        import json
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        results = []
        # Create a moving object (e.g., a car)
        for frame_idx in range(0, total_frames, args.sample_rate):
            # Move diagonally
            progress = frame_idx / total_frames
            cx = int(width * (0.2 + 0.6 * progress))
            cy = int(height * (0.5))
            radius = 50
            
            # Create polygon (octagon)
            points = []
            for i in range(8):
                angle = 2 * np.pi * i / 8
                px = cx + radius * np.cos(angle)
                py = cy + radius * np.sin(angle)
                points.append([px/width*100, py/height*100])
            
            results.append({
                "value": {
                    "frame": frame_idx,
                    "points": points,
                    "polygonlabels": ["汽车"],
                    "x": 0, "y": 0, "width": 100, "height": 100 # Required by LS but ignored for polygons usually
                },
                "from_name": "label", "to_name": "video", "type": "videorectangle"
            })
            
        mock_data = [{"predictions": [{"result": results}]}]
        with open(json_output, 'w') as f:
            json.dump(mock_data, f, indent=2)
        
        cmd = "true" # No-op command since we generated file manually

    else:
        # Use MLLM script
        cmd = f"{sys.executable} scripts/video_auto_labeling.py {video_path} --provider {args.provider} --sample-rate {args.sample_rate} --output {json_output}"
    
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    
    if ret != 0 or not json_output.exists():
        print("❌ Labeling failed.")
        return
        
    print(f"✓ Labels saved to: {json_output}")
    
    # 2. Generate Video
    print("\n[Step 2/2] Generating Visualized Video...")
    
    # Import the function from create_annotated_videos_batch
    # We need to add the current directory to path to import from scripts if needed, 
    # but create_annotated_videos_batch is in root.
    sys.path.append(os.getcwd())
    try:
        from create_annotated_videos_batch import create_annotated_video
        
        success = create_annotated_video(str(video_path), str(json_output), str(video_output))
        
        if success:
            print(f"✓ Annotated video saved to: {video_output}")
            print("\n✅ Pipeline Complete!")
            print(f"1. JSON Data: {json_output}")
            print(f"2. Video:     {video_output}")
        else:
            print("❌ Video generation failed.")
            
    except ImportError:
        print("❌ Could not import create_annotated_video function.")
        print("Make sure create_annotated_videos_batch.py is in the current directory.")

if __name__ == "__main__":
    main()
