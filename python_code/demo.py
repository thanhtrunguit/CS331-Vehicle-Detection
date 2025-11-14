import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

# Load your trained YOLO model (update the path to your actual .pt file)
model = YOLO("/Users/thanhtrung/UIT/BKAI/Traffic Vehicle Detection/Traffic_Vehicle_Detection/models/yolov9c_e30_cygan_18.pt")


def predict_both(input_image, conf=0.5, show_labels=True):
    """
    Runs YOLO inference on the uploaded image twice:
      1) Raw predictions (no WBF), draws bounding boxes (and optional labels).
      2) Weighted Box Fusion (WBF) predictions, draws fused boxes (and optional labels).
    Both sets of boxes are drawn in green. If show_labels=False, only boxes are drawn.
    Returns:
      - annotated_raw_rgb: numpy array of the image with raw YOLO boxes
      - annotated_wbf_rgb: numpy array of the image with WBF boxes
      - labels_text: a string listing detected classes & confidences if show_labels=True, else empty
    """

    # Copy and convert to BGR for drawing
    img_rgb = input_image.copy()
    img_height, img_width = img_rgb.shape[:2]
    img_bgr_for_raw = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr_for_wbf = img_bgr_for_raw.copy()

    # Run inference (Ultralytics caches internally)
    results = model.predict(img_rgb, conf=conf, verbose=False)

    # === 1) RAW PREDICTIONS ===
    raw_labels = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            confidence = float(box.conf)
            raw_labels.append(f"{class_name} ({confidence:.2f})")

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Draw raw box (green)
            cv2.rectangle(img_bgr_for_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label text only if requested
            if show_labels:
                label_text = f"{class_name}: {confidence:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img_bgr_for_raw,
                    (x1, y1 - text_h - baseline),
                    (x1 + text_w, y1),
                    (0, 255, 0),
                    thickness=cv2.FILLED
                )
                cv2.putText(
                    img_bgr_for_raw,
                    label_text,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

    if not raw_labels:
        raw_labels = ["No objects detected (raw)"]

    # === 2) WEIGHTED BOX FUSION ===
    boxes_list = []
    scores_list = []
    labels_list = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            # Normalize to [0,1] for WBF
            norm_box = [x1 / img_width, y1 / img_height, x2 / img_width, y2 / img_height]
            boxes_list.append(norm_box)
            scores_list.append(float(box.conf))
            labels_list.append(int(box.cls))

    wbf_labels = []
    if boxes_list:
        boxes_fused, scores_fused, labels_fused = weighted_boxes_fusion(
            [boxes_list], [scores_list], [labels_list],
            weights=None, iou_thr=0.1, skip_box_thr=0.0
        )
        for box, score, cls_id in zip(boxes_fused, scores_fused, labels_fused):
            x1 = int(box[0] * img_width)
            y1 = int(box[1] * img_height)
            x2 = int(box[2] * img_width)
            y2 = int(box[3] * img_height)
            class_name = model.names[int(cls_id)] if int(cls_id) < len(model.names) else f"Class {int(cls_id)}"
            wbf_labels.append(f"{class_name} ({score:.2f})")

            cv2.rectangle(img_bgr_for_wbf, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if show_labels:
                label_text = f"{class_name}: {score:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    img_bgr_for_wbf,
                    (x1, y1 - text_h - baseline),
                    (x1 + text_w, y1),
                    (0, 255, 0),
                    thickness=cv2.FILLED
                )
                cv2.putText(
                    img_bgr_for_wbf,
                    label_text,
                    (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
    else:
        wbf_labels = ["No objects detected (WBF)"]

    # Convert BGR back to RGB for Gradio
    annotated_raw_rgb = cv2.cvtColor(img_bgr_for_raw, cv2.COLOR_BGR2RGB)
    annotated_wbf_rgb = cv2.cvtColor(img_bgr_for_wbf, cv2.COLOR_BGR2RGB)

    # Prepare the labels text output
    if show_labels:
        labels_text = (
            "Raw Detections:\n" + "\n".join(raw_labels)
            + "\n\nWBF Detections:\n" + "\n".join(wbf_labels)
        )
    else:
        labels_text = ""

    return annotated_raw_rgb, annotated_wbf_rgb, labels_text


# Build Gradio interface
title = "YOLO Vehicle Detection: Raw vs. WBF (Toggle Labels)"
description = """
Upload an image to see two visualizations in green:

1. **Raw YOLO Predictions**  
2. **Weighted Box Fusion (WBF) Predictions**  

Use the slider to set the confidence threshold.  
Check or uncheck “Show Labels” to toggle overlaying text labels and the label list below.
"""

iface = gr.Interface(
    fn=predict_both,
    inputs=[
        gr.Image(type="numpy", label="Input Image"),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.05, label="Confidence Threshold"),
        gr.Checkbox(label="Show Labels", value=True)
    ],
    outputs=[
        gr.Image(type="numpy", label="Raw Prediction (Green)"),
        gr.Image(type="numpy", label="WBF Prediction (Green)"),
        gr.Textbox(label="Detected Classes & Confidences")
    ],
    title=title,
    description=description,
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
