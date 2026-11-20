import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
import numpy as np

# 1. Device Configuration (Uses GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# 2. Data Augmentation & Normalization
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = os.path.join('data', 'chest_xray')
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}

# num_workers=0 prevents multiprocessing errors on Windows
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
    'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=0)
}

# Handle class imbalance
train_targets = image_datasets['train'].targets
class_counts = np.bincount(train_targets)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = (class_weights / class_weights.sum()).to(device)

# 3. Initialize Lightweight MobileNetV3-Small
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 4. Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(image_datasets['train'])
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")

# Save baseline FP32 model
os.makedirs("models", exist_ok=True)
fp32_path = os.path.join("models", "mobilenetv3_fp32.pth")
torch.save(model.state_dict(), fp32_path)

# 5. Apply 8-Bit Dynamic Quantization on CPU
model_cpu = copy.deepcopy(model).to("cpu")
model_cpu.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu, {nn.Linear}, dtype=torch.qint8
)
int8_path = os.path.join("models", "mobilenetv3_int8.pth")
torch.save(quantized_model.state_dict(), int8_path)

# 6. Clinical & Edge Benchmarking Function
def evaluate_edge_model(eval_model, model_path, name):
    eval_model.eval()
    all_preds, all_probs, all_labels = [], [], []
    latencies = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to("cpu")
            
            start_time = time.time()
            outputs = eval_model(inputs)
            end_time = time.time()
            
            batch_latency_ms = ((end_time - start_time) / inputs.size(0)) * 1000
            latencies.append(batch_latency_ms)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds) * 100
    sens = recall_score(all_labels, all_preds) * 100
    auroc = roc_auc_score(all_labels, all_probs)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    avg_latency = np.mean(latencies)
    
    print(f"\n--- {name} ---")
    print(f"Model Size:      {size_mb:.2f} MB")
    print(f"CPU Latency:     {avg_latency:.2f} ms / scan")
    print(f"Test Accuracy:   {acc:.2f}%")
    print(f"Sensitivity:     {sens:.2f}%")
    print(f"ROC-AUC Score:   {auroc:.4f}")

evaluate_edge_model(model_cpu, fp32_path, "Baseline FP32 Model")
evaluate_edge_model(quantized_model, int8_path, "Quantized INT8 Edge Model")
