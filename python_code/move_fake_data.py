from pathlib import Path
import shutil

# Define source and destination directories
source_images_dir = Path("/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/data_with_cycle_gan/images/train")
dest_images_dir = Path("/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset_structured_sampled_100/images/train")
source_labels_dir = Path("/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/data_with_cycle_gan/labels/train")
dest_labels_dir = Path("/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset_structured_sampled_100/labels/train")

# Ensure destination directories exist
dest_images_dir.mkdir(parents=True, exist_ok=True)
dest_labels_dir.mkdir(parents=True, exist_ok=True)

# Move image files containing "fake" in their names
image_count = 0
for image_path in source_images_dir.glob("*.jpg"):  # Adjust extension if needed (e.g., *.png)
    if "fake" in image_path.name.lower():  # Case-insensitive check for "fake"
        dest_path = dest_images_dir / image_path.name
        shutil.move(str(image_path), str(dest_path))
        print(f"Moved image: {image_path} -> {dest_path}")
        image_count += 1

# Move label files containing "fake" in their names
label_count = 0
for label_path in source_labels_dir.glob("*.txt"):
    if "fake" in label_path.name.lower():  # Case-insensitive check for "fake"
        dest_path = dest_labels_dir / label_path.name
        shutil.move(str(label_path), str(dest_path))
        print(f"Moved label: {label_path} -> {dest_path}")
        label_count += 1

# Summary
print(f"\nSummary:")
print(f"Moved {image_count} image files containing 'fake'.")
print(f"Moved {label_count} label files containing 'fake'.")