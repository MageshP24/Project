from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import yaml

from backend.models.factory import build_model

# -------------------- App Initialization --------------------
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Config & Model Setup --------------------
cfg = yaml.safe_load(open("config.yaml"))
MODEL_TYPE = "ensemble"   # "effnet_b3", "vit_b16" or "ensemble"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Label mapping
idx_to_label = {
    0: "Fake",   # AI-generated
    1: "Real"    # Real image
}

# Load model(s)
if MODEL_TYPE == "ensemble":
    m1 = build_model("effnet_b3", 2).to(device)
    m2 = build_model("vit_b16", 2).to(device)

    # Load weights
    state1 = torch.load("checkpoints/effnet_b3_best.pt", map_location=device)
    state2 = torch.load("checkpoints/vit_b16_best.pt", map_location=device)
    m1.load_state_dict(state1["model"] if "model" in state1 else state1)
    m2.load_state_dict(state2["model"] if "model" in state2 else state2)

    m1.eval()
    m2.eval()
else:
    model = build_model(MODEL_TYPE, 2).to(device)
    state = torch.load(f"checkpoints/{MODEL_TYPE}_best.pt", map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

# Image transform (same as training)
transform = transforms.Compose([
    transforms.Resize((cfg["img_size"], cfg["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------- Prediction Endpoint --------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    # Make prediction
    if MODEL_TYPE == "ensemble":
        with torch.no_grad():
            p1 = torch.softmax(m1(x), dim=1)
            p2 = torch.softmax(m2(x), dim=1)
            probs = (p1 + p2) / 2
            pred = probs.argmax(1).item()
            conf = probs.max().item()
    else:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(1).item()
            conf = probs.max().item()

    label = idx_to_label[pred]
    return {"label": label, "confidence": float(conf * 100)}
