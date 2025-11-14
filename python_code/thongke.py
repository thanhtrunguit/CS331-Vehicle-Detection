import os
from collections import defaultdict

# Define class names
class_names = {
    0: "xe máy",
    1: "xe ô tô con",
    2: "Xe vận tải du lịch (xe khách)",
    3: "Xe vận tải container"
}


# Function to count vehicles in label files
def count_vehicles(labels_dir):
    vehicle_counts = defaultdict(int)

    # Iterate through all subfolders (test, val, train)
    for split in ['test', 'val', 'train']:
        split_dir = os.path.join(labels_dir, split)
        if not os.path.exists(split_dir):
            continue

        # Iterate through all label files
        for label_file in os.listdir(split_dir):
            if label_file.endswith('.txt'):
                file_path = os.path.join(split_dir, label_file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        class_id = int(line.split()[0])  # First number is the class ID
                        vehicle_counts[class_id] += 1

    return vehicle_counts


# Main function to process the dataset and return counts
def analyze_dataset(labels_dir):
    vehicle_counts = count_vehicles(labels_dir)

    print("Phân tích số lượng xe theo nhãn:")
    counts = []
    labels = []
    for class_id in sorted(vehicle_counts.keys()):
        count = vehicle_counts[class_id]
        print(f"Nhãn {class_id}: {class_names[class_id]} - Số lượng: {count}")
        labels.append(class_names[class_id])
        counts.append(count)

    return labels, counts


# Example usage
if __name__ == "__main__":
    labels_dir = '/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset_structured_sampled_100/labels'  # Replace with your labels folder path
    labels, counts = analyze_dataset(labels_dir)
    print("Labels:", labels)
    print("Counts:", counts)