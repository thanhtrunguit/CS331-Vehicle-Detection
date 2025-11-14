import cv2
import gradio as gr
from ultralytics import YOLO
import tempfile
import os

# Load the trained YOLO model (update the path to your .pt file)
model = YOLO("/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/models/yolov9c_e30_cygan_18.pt")

def process_video(video_path, conf=0.5, show_labels=True):
    """
    Process a video file, draw bounding boxes on each frame, and return the path to a temporary video file.
    """
    # Open the input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create a temporary file for the output video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_bgr = frame.copy()  # Keep a BGR copy for drawing

        # Run YOLO inference
        results = model.predict(frame_rgb, conf=conf, verbose=False)

        # Draw bounding boxes on the frame
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Draw bounding box (green)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if show_labels:
                    label_text = f"{class_name}: {confidence:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        frame_bgr,
                        (x1, y1 - text_h - baseline),
                        (x1 + text_w, y1),
                        (0, 255, 0),
                        thickness=cv2.FILLED
                    )
                    cv2.putText(
                        frame_bgr,
                        label_text,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1
                    )

        # Write the annotated frame to the temporary video
        out.write(frame_bgr)

    # Release resources
    cap.release()
    out.release()

    # Return the path to the temporary file
    return temp_file.name

def predict_video(video, conf, show_labels):
    """
    Wrapper function for Gradio to process the video and return the result.
    Deletes the temporary file after processing.
    """
    processed_video_path = process_video(video, conf, show_labels)
    return processed_video_path

# Build Gradio interface
title = "YOLO Vehicle Detection on Video"
description = """
Upload a video to see bounding boxes drawn directly on the frames:
- Bounding boxes are drawn in green.
- Use the slider to set the confidence threshold.
- Check or uncheck “Show Labels” to toggle text labels.
- The processed video is not saved permanently.
"""

iface = gr.Interface(
    fn=predict_video,
    inputs=[
        gr.Video(label="Input Video"),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.05, label="Confidence Threshold"),
        gr.Checkbox(label="Show Labels", value=True)
    ],
    outputs=gr.Video(label="Processed Video"),
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()

    # Clean up temporary files after launching (manual cleanup may be needed)
    for file in os.listdir(tempfile.gettempdir()):
        if file.endswith('.mp4'):
            try:
                os.remove(os.path.join(tempfile.gettempdir(), file))
            except:
                pass