import torch
from torchvision import transforms
from PIL import Image
import os
from model.model import SimpleCNN  # Import mô hình từ model.py
import argparse
import pandas as pd

# 1. Nhận đối số từ dòng lệnh

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default=r'D:\THIEN_PROJECT\cat-dog_classification\dataset\test', help='Path to test image folder')
parser.add_argument('--model_path', type=str, default=r'D:\THIEN_PROJECT\cat-dog_classification\cat_dog_model.pth', help='Path to saved model')
parser.add_argument('--output_csv', type=str, default=r'D:\THIEN_PROJECT\cat-dog_classification\predictions.csv', help='Output CSV file path')
args = parser.parse_args()

# Dùng các giá trị
test_image_dir = args.image_dir
model_path = args.model_path
output_csv_path = args.output_csv

# 2. Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 128

# 3. Transform ảnh test
test_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Load model
model = SimpleCNN(num_classes=2).to(device)

if not os.path.exists(args.model_path):
    raise FileNotFoundError(f"❌ Không tìm thấy file model tại: {args.model_path}")

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# 5. Hàm dự đoán cho 1 ảnh
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = test_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted_class = torch.max(output, 1)

        return predicted_class.item(), probabilities.cpu().numpy()[0]
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ảnh {os.path.basename(image_path)}: {e}")
        return -1, [-1, -1]

# 6. Duyệt folder ảnh và lưu kết quả
predictions = []

for img_name in os.listdir(args.image_dir):
    img_path = os.path.join(args.image_dir, img_name)
    pred_class, probs = predict_image(img_path)

    if pred_class == -1:
        result = {'filename': img_name, 'predicted_class': 'error', 'cat_probability': -1, 'dog_probability': -1}
    else:
        result = {
            'filename': img_name,
            'predicted_class': 'dog' if pred_class == 1 else 'cat',
            'cat_probability': probs[0],
            'dog_probability': probs[1]
        }

    predictions.append(result)
    print(f"📷 {img_name} → {result['predicted_class']} (Cat: {probs[0]:.4f}, Dog: {probs[1]:.4f})" if pred_class != -1 else "")

# 7. Ghi ra CSV
df = pd.DataFrame(predictions)
df.to_csv(args.output_csv, index=False)
print(f"\n✅ Kết quả đã lưu tại: {args.output_csv}")
