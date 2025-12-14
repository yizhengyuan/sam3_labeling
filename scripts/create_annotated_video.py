#!/usr/bin/env python3
"""
Create Annotated Video
Reads a video and a Label Studio JSON file, and produces a video with annotations overlaid.
"""

import cv2
import json
import argparse
import numpy as np
import os
from tqdm import tqdm

def create_annotated_video(video_path, json_path, output_path):
    print(f"Processing video: {video_path}")
    print(f"Loading annotations: {json_path}")
    
    # Load Annotations
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # Handle different JSON structures
    if isinstance(data, list) and len(data) > 0:
        if "predictions" in data[0]:
            results = data[0]['predictions'][0]['result']
        elif "annotations" in data[0]:
            results = data[0]['annotations'][0]['result']
        else:
            results = []
    else:
        results = []
        
    print(f"Found {len(results)} annotation items.")
    
    # Group annotations by frame
    annotations_by_frame = {}
    for item in results:
        val = item.get('value', {})
        
        # Handle sequence (Label Studio Video format)
        if 'sequence' in val:
            labels = val.get('labels', [])
            # Also check specific label types if 'labels' is generic
            if not labels:
                labels = val.get('rectanglelabels', []) or val.get('polygonlabels', [])
                
            for seq_item in val['sequence']:
                frame = seq_item.get('frame')
                if frame is not None:
                    if frame not in annotations_by_frame:
                        annotations_by_frame[frame] = []
                    
                    # Create a flattened item for visualization
                    flat_item = item.copy()
                    flat_item['value'] = seq_item.copy()
                    
                    # Ensure labels are present in the flattened value
                    if labels:
                        # Determine label key based on type or default to rectanglelabels
                        if 'points' in seq_item:
                            flat_item['value']['polygonlabels'] = labels
                        else:
                            flat_item['value']['rectanglelabels'] = labels
                            
                    annotations_by_frame[frame].append(flat_item)
                    
        # Handle flat format (frame in value)
        elif 'frame' in val:
             frame = val['frame']
             if frame not in annotations_by_frame:
                 annotations_by_frame[frame] = []
             annotations_by_frame[frame].append(item)
            
    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Colors for different categories
    colors = {
        "person": (255, 112, 67),      # Orange
        "car": (66, 165, 245),         # Blue
        "motorcycle": (102, 187, 106), # Green
        "bicycle": (255, 193, 7),      # Yellow
        "traffic light": (38, 198, 218), # Cyan
        "traffic sign": (156, 39, 176), # Purple
    }
    
    # Process Frames
    for frame_idx in tqdm(range(total_frames), desc="Rendering Video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get annotations for this frame
        frame_annos = annotations_by_frame.get(frame_idx, [])
        
        # Draw annotations
        for item in frame_annos:
            val = item['value']
            
            # 1. Polygon (Mask)
            if 'points' in val:
                points = val['points']
                label = val.get('polygonlabels', ['unknown'])[0]
                color = colors.get(label, (200, 200, 200))
                
                # Convert normalized points to pixels
                pts = []
                for p in points:
                    px = int(p[0] * width / 100)
                    py = int(p[1] * height / 100)
                    pts.append([px, py])
                
                pts = np.array(pts, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Draw filled polygon with transparency
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                
                # Draw border
                cv2.polylines(frame, [pts], True, color, 2)
                
            # 2. Rectangle (Box)
            elif 'x' in val and 'width' in val:
                x = int(val['x'] * width / 100)
                y = int(val['y'] * height / 100)
                w = int(val['width'] * width / 100)
                h = int(val['height'] * height / 100)
                label = val.get('rectanglelabels', ['unknown'])[0]
                color = colors.get(label, (200, 200, 200))
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        
    cap.release()
    out.release()
    print(f"✓ Saved annotated video to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create Annotated Video")
    parser.add_argument("--video", required=True, help="Path to original video")
    parser.add_argument("--json", required=True, help="Path to Label Studio JSON")
    parser.add_argument("--output", required=True, help="Output video path")
    
    args = parser.parse_args()
    
    create_annotated_video(args.video, args.json, args.output)

if __name__ == "__main__":
    main()
