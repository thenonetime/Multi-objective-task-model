import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 加载特征文件
feature_files = [
    "传统特征250维.xlsx",
    "vgg4096维.xlsx",
    "Clip512维.csv",
    "动态手工6维.csv",
    "c3d487维.csv",
    "timesformer512维.xlsx"
]

# 加载标签文件
label_file = "2187.xlsx"
'''
# 读取特征和标签
features = []
for file in feature_files:
    if file.endswith('.xlsx'):
        features.append(pd.read_excel(file, nrows=100).values)
    elif file.endswith('.csv'):
        features.append(pd.read_csv(file, nrows=100).values)

# 合并所有特征
features = np.concatenate(features, axis=1)

# 读取标签
labels = pd.read_excel(label_file, nrows=100).values


'''
# 读取特征和标签
features = []
for file in feature_files:
    if file.endswith('.xlsx'):
        features.append(pd.read_excel(file).values)
    elif file.endswith('.csv'):
        features.append(pd.read_csv(file).values)

# 合并所有特征
features = np.concatenate(features, axis=1)

# 读取标签
labels = pd.read_excel(label_file).values
# 数据归一化
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
labels = scaler.fit_transform(labels)  # 归一化标签

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)# 替换 NaN 为 0
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)
import torch
import torch.nn as nn
import torch.optim as optim


import torch
import torch.nn as nn

class MultiTaskImprovedModel(nn.Module):
    def __init__(self, input_dim, shared_dim, num_tasks):
        super(MultiTaskImprovedModel, self).__init__()
        # 更深的共享特征提取层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, shared_dim),
            nn.ReLU()
        )
        # 每个任务的注意力机制
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.Tanh(),
                nn.Linear(shared_dim, 1),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])
        # 无任务特定层，直接输出
        self.output_layers = nn.ModuleList([
            nn.Linear(shared_dim, 1) for _ in range(num_tasks)
        ])

    def forward(self, x):
        shared_output = self.shared_layers(x)
        task_outputs = []
        for i in range(len(self.attention_layers)):
            attention_weights = self.attention_layers[i](shared_output)
            attended_features = shared_output * attention_weights
            task_outputs.append(self.output_layers[i](attended_features))
        return torch.cat(task_outputs, dim=1)



# 初始化模型
# 参数设置
input_dim = features.shape[1]
shared_dim = 512
num_tasks = 5

model = MultiTaskImprovedModel(input_dim, shared_dim, num_tasks)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train_model(model, X_train, y_train, X_test, y_test, epochs=100):
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # 检查损失是否为 NaN
        if torch.isnan(loss):
            print("Loss is NaN. Stopping training.")
            break

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 评估模型
        model.eval()
        with torch.no_grad():
            test_inputs = torch.tensor(X_test, dtype=torch.float32)
            test_targets = torch.tensor(y_test, dtype=torch.float32)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_targets)

        train_losses.append(loss.item())
        test_losses.append(test_loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}")

    return train_losses, test_losses


train_losses, test_losses = train_model(model, X_train, y_train, X_test, y_test)

# 保存模型
torch.save(model.state_dict(), "multi_task_model.pth")
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 模型预测
y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

mae_list = []
mse_list = []
r2_list = []

for i in range(num_tasks):
    task_true = y_test[:, i]
    task_pred = y_pred[:, i]

    # 计算MAE, MSE, R²
    mae = mean_absolute_error(task_true, task_pred)
    mse = mean_squared_error(task_true, task_pred)
    r2 = r2_score(task_true, task_pred)

    mae_list.append(mae)
    mse_list.append(mse)
    r2_list.append(r2)

    print(f"Task {i + 1} - MAE: {mae}, MSE: {mse}, R²: {r2}")

print(f"MAE: {mae}, MSE: {mse}, R²: {r2}")
# 绘制损失曲线
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()