# Export trained MobileNetV2 as TorchScript
# Run from notebook directory: python src/infra/runner/export_mobilenetv2.py

import torch
import torch.nn as nn
from torchvision import models

# Reconstruct the model architecture (must match training)
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 2)
)

# Load the trained weights from the ONNX-exported model's source
# Since you don't have a .pth saved, we'll retrace from a fresh model
# You'll need to run this cell in your notebook instead:
#
#   model.cpu()
#   model.eval()
#   scripted = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
#   torch.jit.save(scripted, "/Volumes/T7/models/mobilenet_v2_fp32.pt")
#
# Run the above in your day_6 notebook where `model` is already trained.

print("See comments above — run the export in your day_6 notebook where the trained model is loaded.")
