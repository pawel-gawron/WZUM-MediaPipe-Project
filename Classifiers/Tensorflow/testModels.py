import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import string
import matplotlib.pyplot as plt
import numpy as np

# Tensorflow normal model
model = tf.keras.models.load_model('saved_model/my_model.h5')

def letter_to_int(letter):
    letter = str(letter[0]).lower()
    alphabet = 'abcdefghiklmnopqrstuvwxy'
    if letter in alphabet:
        return float(alphabet.index(letter))  # Dodajemy 1, aby uzyskać liczby od 1 do 24

# Wczytanie danych z pliku XLSX
mediapipe = pd.read_csv('/home/pawel/Documents/RISA/sem1/WZUM/WZUM_2023_DataGatherer/sample_dataset.csv')
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
columns_to_remove = [col for col in X.columns if col.endswith('.z')]

# Erase columns
X = X.drop(columns=columns_to_remove)
X = X.drop(columns=mediapipe.columns[0],axis=1)
# X = X.astype(float)
y = mediapipe['letter'].astype(float)

print(X)
print(y)

# Normalizacja danych
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Ocena skuteczności modelu na zbiorze testowym
test_loss, test_accuracy = model.evaluate(X, y)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')