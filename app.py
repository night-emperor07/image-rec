from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import models, transforms

# Load the pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define labels (ImageNet classes)
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Open the image file
    image = Image.open(file.file).convert("RGB")
    # Preprocess the image
    img_tensor = preprocess(image).unsqueeze(0)
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    predicted_label = labels[predicted.item()]
    return {"prediction": predicted_label}
