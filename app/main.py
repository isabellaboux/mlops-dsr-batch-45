import torch
import io
from pydantic import BaseModel
from torchvision.models import ResNet 
from fastapi import FastAPI, File, UploadFile, Depends
from PIL import Image 
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms 
from app.model import load_model, load_transforms

categories = ["freshapple", "freshbanana", "freshorange", 
              "rottenapple", "rottenbanana", "rottenorange"]

## We must define a __init__.py file in the app folder to make it a package
#from app.model import ...

# We are defining a subclass of BaseModel to structure the response data (our schema)
class Result(BaseModel):
    category: str
    confidence: float
    
    
app = FastAPI()

@app.get('/')
def read_root():
    return {"Message": "API is running. Visit /docs for the Swagger API documentation."}

@app.post('/predict/', response_model=Result)
async def predict(
    input_image: UploadFile = File(...),
    model: ResNet = Depends(load_model),
    transforms: transforms.Compose = Depends(load_transforms)       
    
    ) -> Result:
    
    image = Image.open(io.BytesIO(await input_image.read())).convert("RGB")
    
    image = transforms(image).reshape(1, 3, 224, 224) # Add batch dimension
    
    # turn off Dropout and BatchNorm uses stats from training
    model.eval()
    # This turns off gradient tracking for inference (we don't need gradients for inference)
    with torch.inference_mode():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        predicted_confidence, predicted_class_idx = torch.max(probs, dim=1)
        predicted_category= categories[predicted_class_idx.item()]
    
    return Result(category=predicted_category, 
                  confidence=predicted_confidence.item())