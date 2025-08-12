from ultralytics import YOLO

model = YOLO('MADMbest2.pt')
print(model.names)