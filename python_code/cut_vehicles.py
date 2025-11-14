import os
import cv2
import numpy as np
import shutil
from collections import defaultdict

# Function to convert YOLO format to pixel coordinates
def yolo_to_pixel(bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = map(float, bbox.split())
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return class_id, x_min, y_min, x_max, y_max

# Function to convert pixel coordinates to YOLO format
def pixel_to_yolo(class_id, x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return f"{int(class_id)} {x_center} {y_center} {width} {height}"

# Function to check if two bounding boxes overlap
def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

# Function to augment a target image with vehicles from a source image at original positions
def augment_image(target_img_path, target_label_path, source_img_path, source_label_path, output_img_dir, output_label_dir):
    # Load images
    target_img = cv2.imread(target_img_path)
    source_img = cv2.imread(source_img_path)
    target_height, target_width = target_img.shape[:2]
    source_height, source_width = source_img.shape[:2]

    # Read label files
    with open(target_label_path, 'r') as f:
        target_labels = f.readlines()
    with open(source_label_path, 'r') as f:
        source_labels = f.readlines()

    # Get existing bounding boxes in target image
    target_boxes = []
    for label in target_labels:
        class_id, x_min, y_min, x_max, y_max = yolo_to_pixel(label, target_width, target_height)
        target_boxes.append((x_min, y_min, x_max, y_max))

    # Augment with vehicles from source image at their original positions
    for label in source_labels:
        class_id, x_min_src, y_min_src, x_max_src, y_max_src = yolo_to_pixel(label, source_width, source_height)
        vehicle_patch = source_img[y_min_src:y_max_src, x_min_src:x_max_src]
        patch_height, patch_width = y_max_src - y_min_src, x_max_src - x_min_src

        # Calculate the same relative position in target image
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        x_min_target = int(x_min_src * scale_x)
        y_min_target = int(y_min_src * scale_y)
        x_max_target = int(x_max_src * scale_x)
        y_max_target = int(y_max_src * scale_y)
        new_box = (x_min_target, y_min_target, x_max_target, y_max_target)

        # Adjust box to fit within target image boundaries
        x_min_target = max(0, min(x_min_target, target_width - patch_width))
        y_min_target = max(0, min(y_min_target, target_height - patch_height))
        x_max_target = x_min_target + patch_width
        y_max_target = y_min_target + patch_height
        new_box = (x_min_target, y_min_target, x_max_target, y_max_target)

        # Check for overlap with existing vehicles
        if not any(is_overlapping(new_box, existing_box) for existing_box in target_boxes):
            # Paste the vehicle at the calculated position
            target_img[y_min_target:y_max_target, x_min_target:x_max_target] = vehicle_patch
            target_boxes.append(new_box)
            # Add new label
            new_label = pixel_to_yolo(class_id, x_min_target, y_min_target, x_max_target, y_max_target, target_width, target_height)
            target_labels.append(new_label + '\n')

    # Save augmented image and labels
    output_img_path = os.path.join(output_img_dir, os.path.basename(target_img_path))
    output_label_path = os.path.join(output_label_dir, os.path.basename(target_label_path))
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    cv2.imwrite(output_img_path, target_img)
    with open(output_label_path, 'w') as f:
        f.writelines(target_labels)

# Main function to process the dataset
def process_dataset(dataset_dir, output_dir):
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')

    # Create output directories with original structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(output_labels_dir, split), exist_ok=True)

    # Group files by camera angle
    angle_groups = defaultdict(list)
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(images_dir, split)
        split_labels_dir = os.path.join(labels_dir, split)
        if not os.path.exists(split_images_dir):
            continue
        for img_file in os.listdir(split_images_dir):
            if img_file.endswith('.jpg'):
                angle = '_'.join(img_file.split('_')[:2])  # e.g., cam_01
                label_file = img_file.replace('.jpg', '.txt')
                angle_groups[angle].append((
                    os.path.join(split_images_dir, img_file),
                    os.path.join(split_labels_dir, label_file)
                ))

    # Process each camera angle
    for angle, files in angle_groups.items():
        for i, (target_img_path, target_label_path) in enumerate(files):
            # Select a random source image from the same angle (excluding target)
            source_candidates = [f for f in files if f != (target_img_path, target_label_path)]
            if source_candidates:
                source_img_path, source_label_path = source_candidates[np.random.randint(len(source_candidates))]
                split = os.path.basename(os.path.dirname(target_img_path))
                output_img_dir = os.path.join(output_images_dir, split)
                output_label_dir = os.path.join(output_labels_dir, split)
                augment_image(target_img_path, target_label_path, source_img_path, source_label_path, output_img_dir, output_label_dir)
            else:
                # If no source image available, copy original
                split = os.path.basename(os.path.dirname(target_img_path))
                output_img_dir = os.path.join(output_images_dir, split)
                output_label_dir = os.path.join(output_labels_dir, split)
                shutil.copy(target_img_path, os.path.join(output_img_dir, os.path.basename(target_img_path)))
                shutil.copy(target_label_path, os.path.join(output_label_dir, os.path.basename(target_label_path)))

# Example usage
if __name__ == "__main__":
    dataset_dir = '/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset'
    output_dir = '/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset_structured_copy_paste'
    process_dataset(dataset_dir, output_dir)
    print("Dataset augmentation complete!")