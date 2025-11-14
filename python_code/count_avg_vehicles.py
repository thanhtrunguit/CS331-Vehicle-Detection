import os

# Function to calculate vehicle stats
def calculate_vehicle_stats(labels_dir):
    vehicle_counts_per_image = []

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
                    vehicle_counts_per_image.append(len(lines))  # Count vehicles in this image

    # Calculate max, min, and average
    if vehicle_counts_per_image:
        max_vehicles = max(vehicle_counts_per_image)
        min_vehicles = min(vehicle_counts_per_image)
        avg_vehicles = sum(vehicle_counts_per_image) / len(vehicle_counts_per_image)
    else:
        max_vehicles = 0
        min_vehicles = 0
        avg_vehicles = 0

    return max_vehicles, min_vehicles, avg_vehicles

# Main function to process the dataset and return stats
def analyze_dataset(labels_dir):
    max_vehicles, min_vehicles, avg_vehicles = calculate_vehicle_stats(labels_dir)

    print("Thống kê số lượng xe trên mỗi ảnh:")
    print(f"Số lượng xe tối đa trên một ảnh: {max_vehicles}")
    print(f"Số lượng xe trung bình trên mỗi ảnh: {avg_vehicles:.2f}")

# Example usage
if __name__ == "__main__":
    labels_dir = '/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/dataset/labels'
    analyze_dataset(labels_dir)