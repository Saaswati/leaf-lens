import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── CLASS NAMES ──────────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ── MODEL DEFINITION ─────────────────────────────────────────
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        in_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ── TRANSFORMS ───────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── LAZY GLOBALS (not loaded at import time) ──────────────────
_model    = None
_grad_cam = None

def _load_model():
    """Load model only on first prediction request."""
    global _model, _grad_cam
    if _model is not None:
        return
    device = torch.device("cpu")
    m = PlantDiseaseModel(num_classes=38)
    m.load_state_dict(torch.load("model/best_model.pth", map_location=device))
    m.eval()
    gc.collect()
    _model    = m
    _grad_cam = GradCAM(_model, _model.backbone.conv_head)
    print("✅ Model loaded successfully")

# ── GRAD-CAM ─────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot)
        weights  = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam      = (weights * self.activations).sum(dim=1, keepdim=True)
        cam      = F.relu(cam)
        cam      = F.interpolate(cam, size=(224, 224),
                                 mode='bilinear', align_corners=False)
        cam      = cam.squeeze().numpy()
        cam      = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# ── INFERENCE FUNCTION ───────────────────────────────────────
def predict(image_bytes: bytes):
    _load_model()  # no-op if already loaded

    image      = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs              = _model(img_tensor)
        probs                = torch.softmax(outputs, dim=1)
        confidence, pred_idx = probs.max(dim=1)

    pred_idx   = pred_idx.item()
    confidence = confidence.item() * 100
    class_name = CLASS_NAMES[pred_idx]
    plant      = class_name.split("___")[0].replace("_", " ")
    disease    = class_name.split("___")[1].replace("_", " ")

    # Grad-CAM
    img_tensor2 = transform(image).unsqueeze(0)
    img_tensor2.requires_grad_(True)
    cam_map = _grad_cam.generate(img_tensor2, pred_idx)

    # Visualization
    img_display = np.array(image.resize((224, 224))) / 255.0
    heatmap     = plt.cm.jet(cam_map)[:, :, :3]
    overlay     = np.clip(0.5 * img_display + 0.5 * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("LeafLens Analysis", fontsize=14, fontweight='bold')

    axes[0].imshow(img_display);      axes[0].set_title("Original Image");     axes[0].axis("off")
    axes[1].imshow(cam_map, cmap='jet'); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay);          axes[2].set_title("Overlay");             axes[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    gc.collect()

    return {
        "plant"      : plant,
        "disease"    : disease,
        "confidence" : round(confidence, 2),
        "is_healthy" : "healthy" in disease.lower(),
        "gradcam"    : img_base64
    }