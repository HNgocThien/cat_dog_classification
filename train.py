import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from model.model import SimpleCNN  # Đảm bảo file model.py nằm cùng thư mục hoặc thư mục trong PYTHONPATH

# 1. Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 2. Đường dẫn và cấu hình
base_dir = r"D:\THIEN_PROJECT\cat-dog_classification"
train_data_dir = os.path.join(base_dir, "dataset", "train")       # --> D:\THIEN_PROJECT\cat-dog_classification\dataset\train
train_label_csv = os.path.join(base_dir, "labels", "labels.csv")  # --> D:\THIEN_PROJECT\cat-dog_classification\labels\labels.csv
model_save_path = os.path.join(base_dir, "cat_dog_model.pth")


image_size = 128
batch_size = 32
learning_rate = 0.001
num_epochs = 7
val_size = 0.2

# 3. Dataset tùy chỉnh
class CatDogDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(label_file)
        self.transform = transform
        self.valid_samples = []

        for idx in range(len(self.labels_df)):
            img_name = self.labels_df.iloc[idx, 0]
            img_path = os.path.join(self.image_dir, img_name)
            if os.path.exists(img_path):
                self.valid_samples.append((img_name, int(self.labels_df.iloc[idx, 1])))

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_name, label = self.valid_samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# 4. Transforms
train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomRotation(degrees=20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 5. Dataloader
full_dataset = CatDogDataset(train_data_dir, train_label_csv, transform=train_transforms)

val_size_int = int(val_size * len(full_dataset))
train_size_int = len(full_dataset) - val_size_int
train_dataset, val_dataset = random_split(full_dataset, [train_size_int, val_size_int])
val_dataset.dataset.transform = val_transforms

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # Use 0 for Windows
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

dataloaders = {'train': train_dataloader, 'val': val_dataloader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

print(f"📊 Dataset sizes: {dataset_sizes}")

# 6. Khởi tạo mô hình
model = SimpleCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 7. Hàm train
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, device, model_save_path):
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 30)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct_predictions += torch.sum(preds == labels.data)
                total_samples += labels.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = correct_predictions.double() / total_samples

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                torch.save(model.state_dict(), model_save_path)

    print(f"\n✅ Best validation accuracy: {best_val_acc:.4f}")
    return model

# 8. Bắt đầu huấn luyện
print("🚀 Bắt đầu huấn luyện mô hình...")
trained_model = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs, device, model_save_path)
print(f"✅ Huấn luyện xong! Mô hình đã lưu tại: {model_save_path}")
