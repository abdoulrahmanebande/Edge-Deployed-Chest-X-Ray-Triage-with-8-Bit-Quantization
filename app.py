import io
import base64
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, render_template
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = Flask(__name__)

# 1. Load Baseline FP32 Model (for Grad-CAM gradients) & Quantized INT8 Model (for edge inference)
def load_models():
    base_model = models.mobilenet_v3_small(weights=None)
    num_ftrs = base_model.classifier[3].in_features
    base_model.classifier[3] = nn.Linear(num_ftrs, 2)
    base_model.load_state_dict(torch.load("models/mobilenetv3_fp32.pth", map_location="cpu"))
    base_model.eval()

    quant_model = torch.quantization.quantize_dynamic(
        base_model, {nn.Linear}, dtype=torch.qint8
    )
    quant_model.load_state_dict(torch.load("models/mobilenetv3_int8.pth", map_location="cpu"))
    quant_model.eval()
    return base_model, quant_model

fp32_model, int8_model = load_models()

# Target final convolutional layer of MobileNetV3-Small for Grad-CAM
target_layers = [fp32_model.features[-1]]
cam = GradCAM(model=fp32_model, target_layers=target_layers)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        resized_img = img.resize((224, 224))
        input_tensor = transform(img).unsqueeze(0)

        # Run fast INT8 quantized inference on CPU
        start = time.time()
        with torch.no_grad():
            outputs = int8_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        latency_ms = round((time.time() - start) * 1000, 2)

        pneumonia_prob = round(probs[0][1].item() * 100, 2)
        pred_class = 1 if probs[0][1].item() >= 0.5 else 0
        prediction = "PNEUMONIA" if pred_class == 1 else "NORMAL"

        # Generate Grad-CAM heatmap overlay targeting the predicted class
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        rgb_float = np.array(resized_img, dtype=np.float32) / 255.0
        cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        return render_template(
            "index.html",
            prediction=prediction,
            confidence=pneumonia_prob,
            latency=latency_ms,
            orig_img=to_base64(resized_img),
            cam_img=to_base64(Image.fromarray(cam_overlay))
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
