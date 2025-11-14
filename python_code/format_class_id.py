import os
from pathlib import Path

# Define your folder structure
base_path = Path("/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset/labels")
subfolders = ["train", "val", "test"]

# Define the class ID remapping
id_mapping = {4: 0, 5: 1, 6: 2, 7: 3}

# Iterate through each label subfolder
for subfolder in subfolders:
    folder_path = base_path / subfolder
    for label_file in folder_path.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in id_mapping:
                parts[0] = str(id_mapping[class_id])
                new_lines.append(" ".join(parts))
            elif class_id < 4:
                # Keep 0–3 unchanged
                new_lines.append(line.strip())
            else:
                # Ignore or warn about unknown IDs
                print(f"Warning: Unexpected class_id {class_id} in file {label_file}")

        # Overwrite the file with updated labels
        with open(label_file, "w") as f:
            f.write("\n".join(new_lines) + "\n")
