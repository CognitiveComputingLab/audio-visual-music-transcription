import json
import numpy as np
import cv2
import os

# Load the json file with all the metadata
with open("../../pianoDetectData/train/_annotations.coco.json") as f:
    data = json.load(f)

# Correctly interpret the JSON file
image_info = {img["id"]: {"file_name": img["file_name"], "width": img["width"], "height": img["height"]} for img in data["images"]}
annotations = data["annotations"]

# Saved masks location
masks_dir = "../../pianoDetectData/train_masks"
if not os.path.exists(masks_dir):
    os.makedirs(masks_dir)

# For every annotation in the JSON file
for i in annotations:
    image_id = i["image_id"]
    bbox = i["bbox"]
    image = image_info[image_id]

    # Create a blank image with the same dimensions as the original image
    mask = np.zeros((image["height"], image["width"]), dtype=np.uint8)

    # Fill the area with white/1 to represent the keyboard
    x, y, width, height = map(int, bbox)
    mask[y:y+height, x:x+width] = 1

    # Save the mask image
    mask_file = os.path.splitext(image["file_name"])[0] + "_mask.png"
    # Ensure the masks are visible 
    cv2.imwrite(os.path.join(masks_dir, mask_file), mask * 255)

print("Done")
