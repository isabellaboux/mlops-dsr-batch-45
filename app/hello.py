from loadotenv import load_env
import os
import wandb

load_env() # load the variables from the .env file 
assert 'WANDB_API_KEY' in os.environ, "WANDB_API_KEY not found in environment variables."

# get Wanddb model math
artifact_path = "ib-lowpri-dsr/mlops_dsr_batch_45/resnet18:v0"

# define 
MODELS_DIR = "../models"
MODEL_FILENAME = "best_model.pth"

# if dir des not exist, create it
os.makedirs(MODELS_DIR, exist_ok=True)

wandb.login(key=os.getenv('WANDB_API_KEY'))
api = wandb.Api()

artifact = api.artifact(artifact_path, type="model")
artifact.download(root=MODELS_DIR)