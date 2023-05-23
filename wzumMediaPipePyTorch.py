import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import string
import matplotlib.pyplot as plt
import numpy as np

def letter_to_int(letter):
    letter = str(letter[0]).lower()
    alphabet = string.ascii_lowercase
    if letter in alphabet:
        return float(alphabet.index(letter))  # Dodajemy 1, aby uzyskać liczby od 1 do 26

# Wczytanie danych z pliku XLSX
mediapipe = pd.read_excel('WZUM dataset.xlsx', sheet_name="Main")
    # df = pd.concat(mediapipe)
    # print(mediapipe)

for index, row in mediapipe.iterrows():
    mediapipe.at[index, 'letter'] = letter_to_int([mediapipe.at[index, 'letter']])

# Wyodrębnienie nazw kolumn
columns = mediapipe.columns.tolist()

# Wybór kolumn do usunięcia
# columnsErase = columns[127 : 129+1]
columnsErase = columns[64 : 129+1]

X = mediapipe.drop(columns=columnsErase)

# Columns to erase
# columns_to_remove = [col for col in X.columns if col.endswith('.z')]

# # Erase columns
# X = X.drop(columns=columns_to_remove)
X = X.drop(columns=mediapipe.columns[0],axis=1)
# X.to_excel('saved_file.xlsx', index = False)
# X = X.astype(float)
y = mediapipe['letter'].astype(float)

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                stratify=y,
                                                random_state=0,
                                                test_size=0.2)

# Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Konwersja danych do tensorów PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# Definicja modelu sieci
class TabularNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Inicjalizacja modelu
input_dim = X_train.shape[1]
hidden_dim = 64
output_dim = 24
model = TabularNet(input_dim, hidden_dim, output_dim)

# Definicja funkcji straty i optymalizatora
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trening modelu
num_epochs = 180
batch_size = 10

train_losses = []
test_losses = []

# Obliczenie straty treningowej i testowej oraz dokładności w każdej epoce
for epoch in range(num_epochs):
    model.train()  # Ustawienie modelu w trybie treningowym
    for i in range(0, X_train.shape[0], batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Wyczyszczenie gradientów parametrów
        optimizer.zero_grad()
        
        # Przejście przez model
        outputs = model(batch_X)
        
        # Obliczenie straty
        loss = criterion(outputs, batch_y)
        
        # Propagacja wsteczna i aktualizacja parametrów
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())  # Dodanie straty treningowej do listy

    # Obliczenie straty testowej w każdej epoce
    model.eval()  # Ustawienie modelu w trybie ewaluacji
    with torch.no_grad():
        test_loss = 0.0
        for i in range(0, X_test.shape[0], batch_size):
            batch_X_test = X_test[i:i+batch_size]
            batch_y_test = y_test[i:i+batch_size]
            test_outputs = model(batch_X_test)
            test_loss += criterion(test_outputs, batch_y_test).item()
    
        test_loss /= len(y_test)
        test_losses.append(test_loss)

    # Wydrukowanie straty treningowej w każdej epoce
    print(f'Epoch: {epoch+1}, Training Loss: {train_losses[-1]}')

# Generowanie wykresu straty treningowej i testowej
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Obliczenie dokładności na zbiorze testowym
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.round(test_outputs)
    accuracy = (predictions == y_test).sum().item() / len(y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')