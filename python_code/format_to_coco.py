import os
import json
from PIL import Image


def convert_to_coco(image_dir, label_dir, output_json, classes):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": cls} for i, cls in enumerate(classes, 1)]
    }
    image_id = 1
    annotation_id = 1

    for img_name in os.listdir(image_dir):
        if not img_name.endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(image_dir, img_name)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            continue

        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        label_file = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:  # Ensure correct format
                        print(f"Skipping invalid label in {label_file}: {line.strip()}")
                        continue
                    try:
                        cls_id = int(parts[0])  # Class ID (0, 1, 2, 3)
                        x_center, y_center, w, h = map(float, parts[1:])
                        # Convert normalized to absolute coordinates
                        x = (x_center - w / 2) * width
                        y = (y_center - h / 2) * height
                        w_abs = w * width
                        h_abs = h * height
                        coco["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": cls_id + 1,  # Map 0->1, 1->2, 2->3, 3->4
                            "bbox": [x, y, w_abs, h_abs],
                            "area": w_abs * h_abs,
                            "iscrowd": 0
                        })
                        annotation_id += 1
                    except ValueError as e:
                        print(f"Error parsing line in {label_file}: {line.strip()}, {e}")
                        continue
        image_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco, f)


# Define class names for IDs 0, 1, 2, 3
classes = ['xe máy', 'xe ô tô con', 'xe vận tải du lịch', 'xe vận tải container']

# Paths to your dataset
base_dir = '/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset'  # Replace with your dataset path
for split in ['train', 'val', 'test']:
    image_dir = os.path.join(base_dir, 'images', split)
    label_dir = os.path.join(base_dir, 'labels', split)
    output_json = os.path.join(base_dir, f'{split}_coco.json')
    convert_to_coco(image_dir, label_dir, output_json, classes)