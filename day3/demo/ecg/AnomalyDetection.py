# %%
# Importings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
import torch
# %%
# set random seed for reproducibility
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
# Load data
df = pd.read_csv('ecg.csv', header=None)
raw_data = df.values
df.head()
# %%
# Split sets
labels = raw_data[:, -1]
features = raw_data[:, :-1]
X, test_X, y, test_y = train_test_split(
    features, labels, test_size=0.2, random_state=2
)
train_X, valid_X, train_y, valid_y = train_test_split(
    X, y, test_size=0.2, random_state=2
)
# %%
# Normalization to [0, 1]
scaler = MinMaxScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
valid_X = scaler.transform(valid_X)
test_X = scaler.transform(test_X)
# %% 
# Train the autoencoders with only normal rhythms
train_y = train_y.astype(bool)
test_y = test_y.astype(bool)

normal_train_X = train_X[train_y]
anomalous_train_X = train_X[~train_y]

normal_test_X = test_X[test_y]
anomalous_test_X = test_X[~test_y]
# %% 
# plot a normal ECG
# denormalize the data for visualization 
original_train_normal = scaler.inverse_transform(normal_train_X)

plt.grid()
plt.plot(range(len(original_train_normal[1])), original_train_normal[1])
plt.title("A Normal ECG")
plt.show()
# %% 
# Plot an anomalous ECG
# denormalize the data for visualization 
original_train_anomalous = scaler.inverse_transform(anomalous_train_X)

plt.grid()
plt.plot(range(len(original_train_anomalous[1])), original_train_anomalous[1])
plt.title("An Anomalous ECG")
plt.show()
# %%
# convert to pyTorch tensors
batch_size = 512
from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_tensor = torch.FloatTensor(normal_train_X).to(device)
valid_tensor = torch.FloatTensor(valid_X).to(device)
train_dataset = TensorDataset(train_tensor, train_tensor)
valid_dataset = TensorDataset(valid_tensor, valid_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
# %% 
# define the model
import torch.nn as nn
class AnomalyDetector(nn.Module):
    def __init__(self, input_dim=140):
        super(AnomalyDetector, self).__init__()
        # define the encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # define the decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()  # to ensure output is in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

# instantiate the model
model = AnomalyDetector(input_dim=normal_train_X.shape[1]).to(device)
# %%
# define the loss function and optimizer
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.L1Loss()
# %% 
# train the model
epochs = 20
train_losses = []
valid_losses = []
for epoch in range(epochs):
    # training phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        reconstructed = model(data)
        loss = criterion(reconstructed, data)
        
        loss.backward()
        optimizer.step()

        batch_size = data.size(0)
        train_loss += loss.item() * batch_size
        

    
    avg_train_loss = train_loss / len(train_loader.dataset)
    
    # validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for data, _ in valid_loader:
            reconstructed = model(data)
            loss = criterion(reconstructed, data)

            batch_size = data.size(0)
            val_loss += loss.item() * batch_size
    
    avg_val_loss = val_loss / len(valid_loader.dataset)
    
    # record losses
    train_losses.append(avg_train_loss)
    valid_losses.append(avg_val_loss)
    
    # print progress
    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')


# %% 
# plot the training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss", color='blue')
plt.plot(valid_losses, label="Validation Loss", color='red')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MAE)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final Training Loss: {train_losses[-1]:.6f}")
print(f"Final Validation Loss: {valid_losses[-1]:.6f}")
# %% 
# Plotting reconstruction of normal data from test set
model.eval()
normal_test_X_tensor = torch.FloatTensor(normal_test_X).to(device)
with torch.no_grad():
    normal_decoded_data = model(normal_test_X_tensor).cpu().numpy()

plt.plot(normal_test_X[0], 'b')
plt.plot(normal_decoded_data[0], 'r')
plt.fill_between(np.arange(140), normal_decoded_data[0], normal_test_X[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
# %% 
# Plotting reconstruction of anomalous data from the test set
model.eval()
anomalous_test_X_tensor = torch.FloatTensor(anomalous_test_X).to(device)
with torch.no_grad():
    anomalous_decoded_data = model(anomalous_test_X_tensor).cpu().numpy()

plt.plot(anomalous_test_X[0], 'b')
plt.plot(anomalous_decoded_data[0], 'r')
plt.fill_between(np.arange(140), anomalous_decoded_data[0], anomalous_test_X[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()
# %% 
# detect anomalies
model.eval()
with torch.no_grad():
    reconstructions = model(train_tensor)
    train_loss = torch.mean(torch.abs(reconstructions - train_tensor), dim=1).cpu().numpy()
plt.hist(train_loss, bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()
# %% 
# determining threshold
threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
# %% 
# check anomalies reconstrution distribution on test set
test_tensor = torch.FloatTensor(anomalous_test_X).to(device)
model.eval()
with torch.no_grad():
    reconstructions = model(test_tensor)
    test_loss = torch.mean(torch.abs(reconstructions - test_tensor), dim=1).cpu().numpy()

plt.hist(test_loss, bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()
# %% 
# prediction and evaluation
from sklearn.metrics import roc_curve
import torch.nn.functional as F
def predict(model, data, threshold):
  model.eval()
  data_tensor = torch.FloatTensor(test_X).to(device)
  with torch.no_grad():
    reconstructions = model(data_tensor)
    test_loss = torch.mean(torch.abs(reconstructions - data_tensor), dim=1).cpu().numpy()
  prediction = (test_loss < threshold).astype(int)
  return prediction



prediction = predict(model, test_X, threshold)

fpr, tpr, _ = roc_curve(prediction, test_y)
print("TPR: {}, FPR: {}".format(tpr[1], fpr[1]))
# %%
