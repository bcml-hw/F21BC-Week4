"""
This is a PyTorch implementation of the provided Keras ConvLSTM model.

Attention 1: If LSTM is designed to capture long-term dependencies in time series, 
             Using shuffle will lead to the destruction of time dependence, 
             so we set shuffle=False in DataLoader.
             
Attention 2: The default learning rate of RMSprop in Keras is 0.001, while in PyTorch it is 0.01. 
             We need to set it manually if we use PyTorch version of ConvLSTM.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import data_handler as data
import numpy as np
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Create model class equivalent to the Sequential model
class ConvLSTMModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(ConvLSTMModel, self).__init__()
        timesteps, num_features = input_shape

        self.conv1 = nn.Conv1d(num_features, 128, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool1d(2)
        
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.dense = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch, timesteps, features)
        # Conv1D expects (batch, features, timesteps)
        x = x.transpose(1, 2)
        
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        
        # Transpose back for LSTM: (batch, timesteps, features)
        x = x.transpose(1, 2)
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the last output from LSTM
        x = lstm_out[:, -1, :]
        
        # Apply ReLU activation as in original LSTM
        x = F.relu(x)
        
        # Dense layer with softmax
        x = self.dense(x)
        x = F.softmax(x, dim=1)

        return x

def create_model(shape, num_classes):
    model = ConvLSTMModel(shape, num_classes)
    return model

if __name__ == '__main__':
    batch_size   = 32
    timesteps    = 60
    num_features = 19
    accuracies   = list()    
    
    # 10-fold
    #for fold in range(10):
    for fold in range(1):
        print('initializing fold', fold)
        
        # Create model
        model = create_model(shape = (timesteps, num_features),
                           num_classes = 20)
        model = model.to(device)
        
        # Load data
        X_train, y_train, X_test, y_test = data.load_multimodal(fold+1, window_size = timesteps)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test).to(device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        # Attention: If LSTM is designed to capture long-term dependencies in time series, 
        # Using shuffle will lead to the destruction of time dependence
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Compile equivalent - set optimizer and loss
        # In pytorch, the default RMSprop lr=0.01. We need to set lr manually.
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop (equivalent to model.fit)
        epochs = 150
        model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data_batch, target_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data_batch)
                
                # Convert target from one-hot to class indices if needed
                if target_batch.dim() > 1 and target_batch.shape[1] > 1:
                    target_indices = torch.argmax(target_batch, dim=1)
                else:
                    target_indices = target_batch.long()
                
                # Calculate loss (remove softmax from model for CrossEntropyLoss)
                # Need to modify forward pass to not include softmax when training
                model_output = model.dense(F.relu(model.lstm(
                    F.relu(model.conv2(
                        model.maxpool1(F.relu(model.conv1(data_batch.transpose(1, 2))))
                    )).transpose(1, 2)
                )[0][:, -1, :]))
                
                loss = criterion(model_output, target_indices)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(model_output.data, 1)
                total += target_indices.size(0)
                correct += (predicted == target_indices).sum().item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                acc = 100 * correct / total
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')
        
        # Evaluation (equivalent to model.evaluate)
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            # Convert test target from one-hot to class indices if needed
            if y_test_tensor.dim() > 1 and y_test_tensor.shape[1] > 1:
                y_test_indices = torch.argmax(y_test_tensor, dim=1)
            else:
                y_test_indices = y_test_tensor.long()
            
            # Forward pass on test data
            test_outputs = model.dense(F.relu(model.lstm(
                F.relu(model.conv2(
                    model.maxpool1(F.relu(model.conv1(X_test_tensor.transpose(1, 2))))
                )).transpose(1, 2)
            )[0][:, -1, :]))
            
            loss = criterion(test_outputs, y_test_indices)
            test_loss = loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(test_outputs.data, 1)
            total = y_test_indices.size(0)
            correct = (predicted == y_test_indices).sum().item()
            acc = correct / total
        
        accuracies.append(acc)
        print("Model acc:", acc, "loss:", test_loss)
        
    print('mean acc:', sum(accuracies)/len(accuracies))
