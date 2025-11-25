# Testing SAM 3 Integration

Follow these steps to verify that SAM 3 is working correctly.

## Prerequisites
1.  **GPU**: Ensure you have a GPU available.
2.  **Installation**: Make sure you have installed `sam3` and downloaded a checkpoint (see `docs/SAM3_GUIDE.md`).

## Step 1: Prepare Data
We will use a sample image. You can use any image, but let's assume `data/test_image.jpg`.

## Step 2: Generate Initial Boxes (Optional)
If you want to test the **Box-to-Mask** refinement, first generate boxes using the MLLM script.

```bash
# Generate boxes using Qwen (or other provider)
python scripts/image_auto_labeling.py data/test_image.jpg --provider qwen --output boxes.json
```

## Step 3: Run SAM 3
Now run the SAM 3 script.

### Test A: Box Refinement (Box -> Mask)
```bash
python scripts/sam3_auto_labeling.py data/test_image.jpg \
  --mode box_to_mask \
  --box_input boxes.json \
  --checkpoint checkpoints/sam3_h_4b8939.pth \
  --output masks_from_box.json
```

### Test B: Text to Mask (Text -> Mask)
```bash
python scripts/sam3_auto_labeling.py data/test_image.jpg \
  --mode text_to_mask \
  --text_prompt "car . person" \
  --checkpoint checkpoints/sam3_h_4b8939.pth \
  --output masks_from_text.json
```

## Step 4: Visualize Results
We have updated `scripts/visualize_result.py` to support masks.

```bash
# Visualize the masks
# Note: visualize_result.py is designed for video JSONs usually, but works for image JSONs 
# if they follow the same structure. 
# For this test, we might need to slightly adapt the visualization call if it expects frame numbers.
# However, our scripts generate standard Label Studio JSONs.
```

*Note: The current `visualize_result.py` is optimized for video frames. For single images, you can use the visualization built into `image_auto_labeling.py` for boxes, but for masks, you might want to inspect the JSON or import into Label Studio.*

**Best Verification:** Import `masks_from_box.json` into Label Studio and check if the polygons match the objects perfectly.
