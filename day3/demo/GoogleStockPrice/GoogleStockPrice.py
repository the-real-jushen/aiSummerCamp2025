# %% 
# Importings...
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import torch

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
stock_df = pd.read_csv('Google_Stock_Price.csv')
stock_df.head(10)
# %%
# Describe the data
stock_df.describe()
# %% 
# Check the data
stock_df.info()
# %% 
# Drop unecessary data
stock_df.drop('Date', axis=1, inplace=True)
# %% 
# Remove commas and change dtypes
stock_df.replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

stock_df.astype('float64')
# %% 
# Split training and validation sets
stock_df_train = stock_df[:int(0.6*len(stock_df))]
stock_df_valid = stock_df[int(0.6*len(stock_df)):int(0.8*len(stock_df))]
stock_df_test = stock_df[int(0.8*len(stock_df)):]
# %% 
# Normalization
scaler = MinMaxScaler()
scaler = scaler.fit(stock_df_train)
stock_df_train = scaler.transform(stock_df_train)
stock_df_valid = scaler.transform(stock_df_valid)
stock_df_test =  scaler.transform(stock_df_test)
# %% 
# Split X and y
def split_x_and_y(array, days_used_to_train=7):
    features = list()
    labels = list()

    for i in range(days_used_to_train, len(array)):
        features.append(array[i-days_used_to_train:i, :])
        labels.append(array[i, -1])
    return np.array(features), np.array(labels)

train_X, train_y = split_x_and_y(stock_df_train)
valid_X, valid_y = split_x_and_y(stock_df_valid)
test_X, test_y = split_x_and_y(stock_df_test)

print('Shape of Train X: {} \n Shape of Train y: {}'.format(train_X.shape, train_y.shape))
print(train_X[:5, -1, -1])
print(train_y[:5])
# %%
# convert to pyTorch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X_tensor = torch.FloatTensor(train_X).to(device)
train_y_tensor = torch.FloatTensor(train_y).to(device)
valid_X_tensor = torch.FloatTensor(valid_X).to(device)
valid_y_tensor = torch.FloatTensor(valid_y).to(device)
test_X_tensor = torch.FloatTensor(test_X).to(device)
test_y_tensor = torch.FloatTensor(test_y).to(device)
# %%
# create DataLoaders for training and validation sets
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
valid_dataset = TensorDataset(valid_X_tensor, valid_y_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
# %%
# define the LSTM model
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dense = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM output: (batch_size, seq_len, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # only take the output of the last time step
        output = self.dense(lstm_out[:, -1, :])  # (batch_size, output_size)
        return output

input_size = train_X.shape[2]
model = LSTMModel(input_size=input_size, hidden_size=64, output_size=1).to(device)

# %%
# define the optimizer and loss function
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
# %%
# train the model
epochs = 100
train_losses = []
valid_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y)
        
        batch_size = batch_x.size(0)
        train_loss += loss.item() * batch_size  
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    valid_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            batch_size = batch_x.size(0)
            valid_loss += loss.item() * batch_size
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_valid_loss = valid_loss / len(valid_loader.dataset)
    
    train_losses.append(avg_train_loss)
    valid_losses.append(avg_valid_loss)
    
    print(f'Epoch {epoch+1}/{epochs} | '
          f'Train Loss: {avg_train_loss:.6f} | '
          f'Valid Loss: {avg_valid_loss:.6f}')
# %% 
# evaluate the model on the test set
model.eval()
with torch.no_grad():
    pred_y = model(test_X_tensor).cpu().numpy().squeeze()
# %%
# plotting the predictions against the ground truth
test_y_np = test_y_tensor.cpu().numpy()

plt.figure(figsize=(10, 6))
plt.plot(range(len(pred_y)), pred_y, label='Prediction', alpha=0.8)
plt.plot(range(len(test_y_np)), test_y_np, label='Ground Truth', alpha=0.8)
plt.xlabel('Amount of samples')
plt.ylabel('Prediction')
plt.title('Prediction vs Ground Truth')
plt.legend()
plt.grid(True)
plt.show()
# %%
