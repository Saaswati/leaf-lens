from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import predict

app = FastAPI(title="LeafLens API")

# Allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result      = predict(image_bytes)
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model": "EfficientNet-B3"}