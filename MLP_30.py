import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.nn.functional as F
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.layers import Dense
import sklearn
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import matplotlib.pyplot as plt

# 训练数据读取
df = pd.read_csv('output_data.csv', header=None)
data_list = df.values.tolist()
# 查看前5行数据以验证
for i in range(0,5):
    print(data_list[i])
print(f" data——list shape: {np.array(data_list).shape}") #输出的结果是105120*20
train_0= np.array(data_list)

# 标签数据读取
df_label = pd.read_csv('all_a_array_2.csv', header=None)
data_label = df_label.values.tolist()
original_data = np.array(data_label)
data_label_list = []
for row in original_data:
    # 将每行数据按6个一组进行分割
    groups = [row[i:i+6] for i in range(0, len(row), 6)]
    data_label_list.extend(groups)
new_data = [list(row) for row in data_label_list]
# new_data = []
# for row in new_data0:
#     # 计算每行所有元素的和
#     row_sum = sum(row)
#     new_data.append([row_sum])
for i in range(0,5):
    print(new_data[i])
print(f"New data shape: {np.array(new_data).shape}") # 输出结果是105120*6
label_0 = np.array(new_data)

# 数据分割
X = train_0
Y = label_0
X_train, X_test, y_train, y_test = train_test_split(X , Y, test_size=0.2,shuffle=False)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(20, 128)
        self.layer2 = nn.Linear(128, 64)
        self.predict = nn.Linear(64, 6)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.predict(x)
        return x



MLPmodel = MLP()

# GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)

# 训练数据转化张量
x_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
x_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
train_data = torch.utils.data.TensorDataset(x_train, y_train)
# 看一下训练数据是不是正确读入
print(train_data[0])


# 定义数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=False)

# 优化器定义
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.MSELoss()


# 模型训练及测试
train_loss_all = []
test_loss_all = []
for epoch in range(200):
    train_loss = 0
    train_num = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        train_num += b_x.size(0)
    train_loss_all.append(train_loss / train_num)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_output = model(x_test)
        test_loss = loss_func(test_output, y_test)
        test_loss_all.append(test_loss.item())

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss_all[-1]:.4f}, Test Loss: {test_loss_all[-1]:.4f}")


# 可视化
plt.figure()
plt.plot(train_loss_all, label='train loss')
plt.plot(test_loss_all, label='test loss')
plt.legend()
plt.grid()
plt.xlabel(' MLP_epoch')
plt.ylabel('MSE loss')
plt.show()

