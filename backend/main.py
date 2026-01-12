import sys
import os
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision.transforms import ToTensor
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CIFAKE_ConvCNN_model import CIFAKE_ConvCNN
from CIFAKE_ResNet_50_model import resnet50
from CIFAKE_ConvCNN_tuned_model import CIFAKE_ConvCNN_Tuned

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
models = {}

app = FastAPI(title="CIFAKE AI-EMAGE DETECTION API", version="1.0")

def load_models():
    try:
        cnn = CIFAKE_ConvCNN().to(DEVICE)
        cnn_path = os.path.join("training_results", "conv_cnn", "conv_cnn.pth")
        cnn.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
        cnn.eval()
        models['conv_cnn'] = cnn
        print("ConvCNN loaded successfully.")
    except Exception as e:
        print(f"Error with loading ConvCNN: {e}")

    try:
        tuned_cnn = CIFAKE_ConvCNN_Tuned(n_neurons=256, dropout_p=0.28).to(DEVICE)
        tuned_path = os.path.join("training_results", "conv_cnn_tuner", "conv_cnn_tuner.pth")
        tuned_cnn.load_state_dict(torch.load(tuned_path, map_location=DEVICE))
        tuned_cnn.eval()
        models['tuned_cnn'] = tuned_cnn
        print("Tuned ConvCNN loaded successfully.")
    except Exception as e:
        print(f"Error with loading Tuned ConvCNN: {e}")

    try:
        resnet = resnet50(num_classes=2).to(DEVICE)
        res_path = os.path.join("training_results", "resnet-50", "resnet50.pth")
        resnet.load_state_dict(torch.load(res_path, map_location=DEVICE))
        resnet.eval()
        models['resnet'] = resnet
        print("ResNet-50 loaded successfully.")
    except Exception as e:
        print(f"Error with loading ResNet-50: {e}")


load_models()

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = ToTensor()
    return transform(image).unsqueeze(0).to(DEVICE)

@app.get("/")
def read_root():
    return {"status": "CIFAKE API is running"}

@app.post("/predict/{model_type}")
async def predict(model_type: str, file: UploadFile = File(...)):
    if model_type not in models:
        raise HTTPException(status_code=404, detail="Model not found or failed to load")

    model = models[model_type]
    image_bytes = await file.read()

    try:
        tensor = transform_image(image_bytes)

        with torch.no_grad():
            output = model(tensor)

            if model_type in ["conv_cnn", "tuned_cnn"]:
                prob = output.item()
                prediction = "REAL" if prob > 0.5 else "FAKE"
                confidence = prob if prob > 0.5 else 1 - prob

            elif model_type == "resnet":
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted_class = torch.max(output, 1)

                classes = ['FAKE', 'REAL']
                prediction = classes[predicted_class.item()]
                confidence = probabilities[0][predicted_class.item()].item()

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            "model": model_type,
            "prediction": prediction,
            "confidence": f"{confidence:.4f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))