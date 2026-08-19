"""
Human Detection - Inference Script
====================================
Uses the trained YOLOv8 model to detect humans in images/videos
and draw bounding boxes around them.

Usage:
    python detect.py --source path/to/image_or_video
    python detect.py --source 0           # webcam
    python detect.py --source path/to/folder
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Human Detection using YOLOv8")
    parser.add_argument(
        "--source",
        type=str,
        default="images/val",
        help="Path to image, video, folder, or webcam (0). Default: images/val",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/human_detection/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for NMS (default: 0.45)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results with bounding boxes",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display results in a window",
    )
    args = parser.parse_args()

    # ──────────────────────────────────────────────
    # Load the trained model
    # ──────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # ──────────────────────────────────────────────
    # Run detection
    # ──────────────────────────────────────────────
    print(f"Running detection on: {args.source}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print("-" * 50)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        show=args.show,
        name="human_detection_results",
        line_width=2,
        show_labels=True,
        show_conf=True,
    )

    # ──────────────────────────────────────────────
    # Print detection summary
    # ──────────────────────────────────────────────
    total_detections = 0
    for i, result in enumerate(results):
        num_detections = len(result.boxes)
        total_detections += num_detections
        if num_detections > 0:
            print(f"Image {i + 1}: {num_detections} person(s) detected")
            for j, box in enumerate(result.boxes):
                conf = box.conf[0].item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(
                    f"  └─ Person {j + 1}: "
                    f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), "
                    f"confidence={conf:.2%}"
                )

    print(f"\n{'='*50}")
    print(f"Total images processed: {len(results)}")
    print(f"Total persons detected: {total_detections}")
    print(f"Results saved to: runs/detect/human_detection_results/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
