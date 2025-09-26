"""
LSTM Text Generation using PyTorch

Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import sys
import io
import requests
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Download and load text data (equivalent to get_file)
def get_file_pytorch(filename, origin):
    if not os.path.exists(filename):
        print('Downloading...')
        response = requests.get(origin)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
    return filename

path = get_file_pytorch(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# Convert to PyTorch tensors
x_tensor = torch.FloatTensor(x).to(device)
y_tensor = torch.FloatTensor(y).to(device)

# build the model: a single LSTM
print('Build model...')
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the last output
        out = self.fc(lstm_out[:, -1, :])
        return self.softmax(out)

model = LSTMModel(len(chars), 128, len(chars)).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds.cpu().detach().numpy()).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            x_pred_tensor = torch.FloatTensor(x_pred).to(device)
            model.eval()
            with torch.no_grad():
                preds = model(x_pred_tensor)[0]
            model.train()
            
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# Initial text generation before training
on_epoch_end(0, 0)

# Training loop equivalent to model.fit
batch_size = 128
epochs = 1

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    
    # Shuffle data
    indices = np.random.permutation(len(sentences))
    x_shuffled = x_tensor[indices]
    y_shuffled = y_tensor[indices]
    
    total_loss = 0
    num_batches = 0
    
    # Process in batches
    for i in range(0, len(sentences), batch_size):
        batch_end = min(i + batch_size, len(sentences))
        x_batch = x_shuffled[i:batch_end]
        y_batch = y_shuffled[i:batch_end]
        
        # Convert y_batch from one-hot to class indices
        y_batch_indices = torch.argmax(y_batch, dim=1)
        
        optimizer.zero_grad()
        
        outputs = model(x_batch)
        # Remove softmax from model output for CrossEntropyLoss
        outputs = model.fc(model.lstm(x_batch)[0][:, -1, :])
        
        loss = criterion(outputs, y_batch_indices)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if i // batch_size % 10 == 0:
            print(f'Batch {i//batch_size}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / num_batches
    print(f'Average loss: {avg_loss:.4f}')
    
    # Generate text at end of epoch
    on_epoch_end(epoch + 1, 0)