import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from model.model import SimpleCNN  # Đảm bảo đúng đường dẫn tới model.py

# 1. Cấu hình model và device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'D:\THIEN_PROJECT\cat-dog_classification\cat_dog_model.pth'
image_size = 128

# 2. Load model
model = SimpleCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3. Transform cho ảnh test
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Hàm dự đoán
def predict(image: Image.Image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(probs)
    return pred_class, probs

# 5. Giao diện Streamlit
st.set_page_config(page_title="🐱🐶 Cat vs Dog Classifier", layout="centered")

st.title("🐾 Dự đoán Mèo hay Chó")
st.write("Tải lên một ảnh để phân loại là **mèo** hay **chó** bằng mô hình học sâu của bạn.")

uploaded_file = st.file_uploader("📤 Tải ảnh lên", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh bạn đã tải lên", use_column_width=True)

    pred_class, probs = predict(image)
    label = "🐶 Chó" if pred_class == 1 else "🐱 Mèo"
    st.markdown(f"### ✅ Kết quả: **{label}**")
    st.markdown(f"- Xác suất Mèo: `{probs[0]:.4f}`")
    st.markdown(f"- Xác suất Chó: `{probs[1]:.4f}`")
