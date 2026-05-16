from ultralytics import YOLO

model = YOLO("yolov10n.pt")  # uses the model already in your repo

results = model.train(
    data    = "dataset/dataset.yaml",
    epochs  = 50,
    imgsz   = 640,
    batch   = 8,
    name    = "barrels_v1",
    device  = "cpu",
    patience= 15,
    lr0     = 0.01,
    lrf     = 0.01,
)
print("Best weights saved to:", results.save_dir)
print("Copy runs/detect/barrels_v1/weights/best.pt → barrels.pt in your project root")