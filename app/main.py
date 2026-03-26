import torch
import io
from pydantic import BaseModel
from torchvision.models import ResNet 
from fastapi import FastAPI, File, UploadFile, Depends
from PIL import Image 
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms 
from app.model import load_model, load_transforms


# List of class names for model predictions.
# Must match the order used during model training.
categories = ["freshapple", "freshbanana", "freshorange", 
              "rottenapple", "rottenbanana", "rottenorange"]

# Define response schema with Pydantic.
# FastAPI uses this to validate and serialize API outputs.
class Result(BaseModel):
    category: str
    confidence: float
    
# Create FastAPI app instance
app = FastAPI()

# Health check endpoint
@app.get('/')
def read_root():
    return {"Message": "API is running. Visit /docs for the Swagger API documentation."}

# Prediction endpoint
@app.post('/predict/', response_model=Result)
async def predict(
    input_image: UploadFile = File(...),
    model: ResNet = Depends(load_model),
    transforms: transforms.Compose = Depends(load_transforms)       
    ) -> Result:
    
    # Convert bytes to PIL image and ensure RGB color mode
    image = Image.open(io.BytesIO(await input_image.read())).convert("RGB")
    
    # Add batch dimension for model and apply preprocessing transforms (resize, normalize, etc.)
    image = transforms(image).reshape(1, 3, 224, 224) # Add batch dimension
    
    # Set model to evaluation mode (disable dropout, use running batch-norm stats)
    model.eval()

    # Disable gradients during prediction to save memory and improve speed
    with torch.inference_mode():
        logits = model(image)

         # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Get highest probability and index (in our categories list)
        predicted_confidence, predicted_class_idx = torch.max(probs, dim=1)
        predicted_category= categories[predicted_class_idx.item()]
    
    # Return a structured response
    return Result(category=predicted_category, 
                  confidence=predicted_confidence.item())