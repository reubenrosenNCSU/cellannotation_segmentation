from ultralytics import YOLO
import os

model = YOLO('snapshots/best.pt')
#model.train(data='coco.yaml', epochs=10)

# for param in model.model.model[0:5].parameters():
#     param.requires_grad = False

model.train(cfg='train_config.yaml')