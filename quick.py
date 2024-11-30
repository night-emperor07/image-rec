import torch
from torchvision import models, transforms

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])