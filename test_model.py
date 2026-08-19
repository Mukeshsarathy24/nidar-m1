"""
Quick Test - Human Detection with YOLOv8 (Pre-trained)
======================================================
Uses the pre-trained YOLOv8n model (COCO class 0 = person)
to test human detection on validation images before custom training.
"""

from ultralytics import YOLO
import os
import glob


def main():
    print("=" * 60)
    print("  YOLOv8 Human Detection - Quick Test")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # 1. Load pre-trained YOLOv8n model
    # ──────────────────────────────────────────────
    print("\n[1/3] Loading pre-trained YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    print("      Model loaded successfully!")

    # ──────────────────────────────────────────────
    # 2. Pick a few test images from validation set
    # ──────────────────────────────────────────────
    val_dir = "images/val"
    test_images = glob.glob(os.path.join(val_dir, "*.jpg"))[:5]
    print(f"\n[2/3] Testing on {len(test_images)} sample images from {val_dir}/")

    # ──────────────────────────────────────────────
    # 3. Run detection (only class 0 = person)
    # ──────────────────────────────────────────────
    print("\n[3/3] Running human detection...\n")
    results = model.predict(
        source=test_images,
        conf=0.25,
        classes=[0],              # Only detect class 0 (person) from COCO
        save=True,
        name="test_results",
        line_width=2,
        show_labels=True,
        show_conf=True,
        device="cpu",
    )

    # ──────────────────────────────────────────────
    # Print results summary
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Detection Results")
    print("=" * 60)

    total_persons = 0
    for i, result in enumerate(results):
        img_name = os.path.basename(result.path)
        num_persons = len(result.boxes)
        total_persons += num_persons
        print(f"\n  Image: {img_name}")
        print(f"  Persons detected: {num_persons}")
        for j, box in enumerate(result.boxes):
            conf = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            print(
                f"    Person {j+1}: "
                f"bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                f"size={w:.0f}x{h:.0f} "
                f"conf={conf:.1%}"
            )

    print(f"\n{'=' * 60}")
    print(f"  Total persons detected: {total_persons} across {len(results)} images")
    print(f"  Results saved to: runs/detect/test_results/")
    print(f"{'=' * 60}")
    print(f"\n  Images with bounding boxes have been saved!")
    print(f"  Open the folder above to see detection results.\n")


if __name__ == "__main__":
    main()
