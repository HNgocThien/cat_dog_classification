import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(64 * 32 * 32, num_classes)  # Giả định input image size = 128x128

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Kiểm tra nhanh mô hình
    model = SimpleCNN(num_classes=2)
    print(model)

    # Tạo một tensor input giả để test forward pass
    dummy_input = torch.randn(1, 3, 128, 128)  # batch_size=1, channels=3, height=128, width=128
    output = model(dummy_input)
    print("Output shape:", output.shape)
