# computer-vision-pipeline/object_detection.py

import torch
from torchvision import models, transforms
from PIL import Image

# Load a pre-trained object detection model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Image preprocessing
transform = transforms.Compose([transforms.ToTensor()])
image = Image.open("path_to_image.jpg")
image_tensor = transform(image).unsqueeze(0)

# Object detection
with torch.no_grad():
    prediction = model(image_tensor)

print(prediction)
