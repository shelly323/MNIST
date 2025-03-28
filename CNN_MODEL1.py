import torch
from torch import nn
import torchvision as TV
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

# 下載 MNIST 數據集
train_data = TV.datasets.MNIST("MNIST/", train=True, transform=None, target_transform=None, download=True)
test_data = TV.datasets.MNIST("MNIST/", train=False, transform=None, target_transform=None, download=True)

print('Number of samples in train_data:', len(train_data))
print('Number of samples in test_data:', len(test_data))

# 定義 CNN 模型
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2))
        self.lin1 = nn.Linear(256 * 5 * 5, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = nn.functional.relu(x)
        
        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = nn.functional.relu(x)
        
        x = x.flatten(start_dim=1)
        x = self.lin1(x)
        x = nn.functional.relu(x)
        x = self.out(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x

# 圖片預處理函數
def prepare_images(xt):
    return xt.unsqueeze(1).float()  # 加入通道軸並轉為 float 型別

# 建立 CNN 模型
model = CNN()
epochs = 100
batch_size = 500
lr = 1e-3
opt = torch.optim.Adam(params=model.parameters(), lr=lr)
lossfn = nn.NLLLoss()

losses = []
acc_CNN = []

# 訓練模型
for i in range(epochs):
    clear_output(wait=True)
    
    opt.zero_grad()
    batch_ids_CNN = np.random.randint(0, 60000, size=batch_size)
    
    xt = train_data.data[batch_ids_CNN]
    xt = prepare_images(xt)
    
    yt = train_data.targets[batch_ids_CNN]  # 修正 train_labels -> targets
    
    pred = model(xt)
    pred_labels = torch.argmax(pred, dim=1)
    
    acc_ = 100.0 * (pred_labels == yt).sum() / batch_size
    acc_CNN.append(acc_.item())
    print("accuracy: ", acc_)
    
    loss = lossfn(pred, yt)
    losses.append(loss.detach().cpu().numpy())  # 轉為 NumPy
    
    loss.backward()
    opt.step()

# 轉為 NumPy 陣列
acc_CNN = np.array(acc_CNN)
losses = np.array(losses)

# 繪製 Loss 曲線
plt.figure(figsize=(10, 7))
plt.xlabel("Training Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.plot(losses, label="Training Loss", color="blue")
plt.legend()
plt.show()

# 繪製 Accuracy 曲線
plt.figure(figsize=(10, 7))
plt.xlabel("Training Epochs", fontsize=16)
plt.ylabel("Training Accuracy", fontsize=16)
plt.plot(acc_CNN, label="Training Accuracy", color="red")
plt.legend()
plt.show()

# 測試模型
test_id = np.random.randint(0, 10000, size=10)
xt = test_data.data[test_id]
xt = prepare_images(xt)
preds = model(xt)
pred_ind = torch.argmax(preds.detach(), dim=1).numpy()

# 測試準確率函數
def test_acc(model):
    xt = prepare_images(test_data.data)  # 移除 .detach()
    yt = test_data.targets
    preds = model(xt)
    pred_ind = torch.argmax(preds.detach(), dim=1)
    acc = (pred_ind == yt).sum().float() / len(test_data)
    return acc

# 計算測試準確率
acc2 = test_acc(model)
print(f'Testing accuracy = {acc2.item() * 100:.2f}%')
