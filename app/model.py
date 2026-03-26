import os
import wandb
import torch
from loadotenv import load_env

from torchvision.models import resnet18, ResNet
from torch import nn
from pathlib import Path
from torchvision.transforms import v2 as transforms

MODELS_DIR = "models"
MODEL_FILENAME = 'best_model.pth'
os.makedirs(MODELS_DIR, exist_ok=True)

def download_artifact():
    '''Download teh model weights from wandb.'''
    load_env() # this loads the variables in the .env file into the environment variables
    assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables"
    
    wandb.login(key=os.getenv('WANDB_API_KEY'))    
    api = wandb.Api()

    artifact_path = f"{os.getenv('WANDB_ORG')}/{os.getenv('WANDB_PROJECT')}/{os.getenv('WANDB_MODEL_NAME')}:{os.getenv('WANDB_MODEL_VERSION')}"
    artifact = api.artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)

def get_raw_model() -> ResNet:
    '''Retursns aa ResNet model with the final fully connected layer modified for 6 classes.'''
    architecture = resnet18(weights=None) # this initilaizes the model with random weights - note we don't need teh ResNet weight, we have the ones fo our fine-tuned model.
    architecture.fc = nn.Sequential(
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=6)
    )

    return architecture

def load_model() -> ResNet:
    '''Downloads the model artifact from Weights & Biases and loads the model weights into a ResNet18 architecture.'''
    download_artifact()
    model_path = Path(MODELS_DIR) / MODEL_FILENAME
    model = get_raw_model()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set the model to evaluation mode
    return model

def load_transforms() -> transforms.Compose:
    '''Load model transformations.'''
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ]
    )
