from ultralytics import YOLO
import os
import torch
import torchvision
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())
print("Torchvision Version:", torchvision.__version__)

# Check if NMS can run on CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = YOLO("yolov8m.pt")  
model.train(data=os.path.join("ds", "data.yaml"), imgsz=640, epochs=100)
model.save('custom_model.pt')
