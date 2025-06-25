# %%
# load data
import pandas as pd
from IPython.display import display
import numpy as np

red_wine = pd.read_csv('red-wine.csv')

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# %%
# Scale to [0, 1]
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# %%
# Split features and target
# convert to numpy array
X_train = df_train.drop('quality', axis=1).values
X_valid = df_valid.drop('quality', axis=1).values
y_train = df_train['quality'].values
y_valid = df_valid['quality'].values
# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).reshape(-1, 1)

batch_size = 256
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(X_valid_tensor, y_valid_tensor), batch_size=batch_size, shuffle=False)

# %%
# set the random seed for reproducibility
import random  
import os
def set_all_seeds(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(1)
# %%
# EarlyStopping
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_state)
# %%
# build a neural network model
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(11, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.seq(x)
    
# instantiate the model
model = MyModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# %%
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.L1Loss()
early_stopping = EarlyStopping(patience=20, min_delta=1e-5)
num_epochs = 500
history = {'loss': [], 'val_loss': []}
# %%
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='D:/temp/tensorboard_logs_redwine')
# %%
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    train_loss = np.mean(train_losses)

    model.eval()
    with torch.no_grad():
        val_losses = []
        for xb, yb in valid_loader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            loss = criterion(output, yb)
            val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

    history['loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    # 写入TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)

    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print(f"Early stopped at epoch {epoch+1}")
        break

history_df = pd.DataFrame(history)
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
writer.close()
# %%
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.figure(figsize=(8, 5))
plt.plot(history_df['loss'], label='Train Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
# %%
