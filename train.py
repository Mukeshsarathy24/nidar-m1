"""
Human Detection Model - YOLOv8 Training Script
================================================
Trains a YOLOv8 model to detect humans with bounding boxes.

Usage:
    python train.py
"""

from ultralytics import YOLO


def main():
    # ──────────────────────────────────────────────
    # 1. Load a pre-trained YOLOv8 nano model
    #    (use 'yolov8s.pt' or 'yolov8m.pt' for better accuracy)
    # ──────────────────────────────────────────────
    model = YOLO("yolov8n.pt")

    # ──────────────────────────────────────────────
    # 2. Train the model on your human dataset
    # ──────────────────────────────────────────────
    results = model.train(
        data="data.yaml",          # Dataset configuration
        epochs=50,                 # Number of training epochs
        imgsz=640,                 # Input image size
        batch=16,                  # Batch size (reduce if GPU memory is low)
        name="human_detection",    # Experiment name
        patience=10,               # Early stopping patience
        save=True,                 # Save checkpoints
        save_period=10,            # Save checkpoint every N epochs
        plots=True,                # Generate training plots
        device="cpu",              # Using CPU (change to 0 for GPU)
        workers=4,                 # Number of data loading workers
        lr0=0.01,                  # Initial learning rate
        lrf=0.01,                  # Final learning rate factor
        optimizer="auto",          # Optimizer (auto selects best)
        augment=True,              # Enable augmentation
        val=True,                  # Run validation during training
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    # ──────────────────────────────────────────────
    # 3. Validate the best model
    # ──────────────────────────────────────────────
    best_model = YOLO("runs/detect/human_detection/weights/best.pt")
    metrics = best_model.val(data="data.yaml")

    print(f"\n{'='*60}")
    print(f"Validation Results:")
    print(f"{'='*60}")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print(f"{'='*60}")

    # ──────────────────────────────────────────────
    # 4. Export the model to ONNX format (optional)
    # ──────────────────────────────────────────────
    best_model.export(format="onnx")
    print("\nModel exported to ONNX format!")
    print(f"Best weights saved at: runs/detect/human_detection/weights/best.pt")


if __name__ == "__main__":
    main()
