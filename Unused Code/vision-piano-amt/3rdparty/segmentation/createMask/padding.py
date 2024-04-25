import cv2
import os

# Padd the few smaller images
def pad_to_target(image, target_width=640, target_height=360, pad_value=0):

    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    
    # Work out the amount to pad by
    new_width = int(width * scale)
    new_height = int(height * scale)
    image_resized = cv2.resize(image, (new_width, new_height))

    pad_width = (target_width - new_width) // 2
    pad_height = (target_height - new_height) // 2

    # Apply padding
    image_padded = cv2.copyMakeBorder(image_resized, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=pad_value)
    return image_padded

img_dir = "../../pianoDetectData/train/masks"

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

# Iterate through the original images
for image_name in os.listdir(img_dir):
    image_path = os.path.join(img_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        continue  # Skip invalid images

    # Pad and resize the image
    image_padded = pad_to_target(image, 640, 360, pad_value=(0, 0, 0))  # Assuming RGB images; adjust the pad_value if needed

    # Save the processed image
    cv2.imwrite(os.path.join(img_dir, image_name), image_padded)

print("DONE")