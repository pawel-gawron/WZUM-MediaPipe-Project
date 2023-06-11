import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import string
import matplotlib.pyplot as plt
import numpy as np

def letter_to_int(letter):
    letter = str(letter[0]).lower()
    alphabet = 'abcdefghiklmnopqrstuvwxy'
    if letter in alphabet:
        return float(alphabet.index(letter))  # Dodajemy 1, aby uzyskać liczby od 1 do 24

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
columns_to_remove = [col for col in X.columns if col.endswith('.z')]

# Erase columns
X = X.drop(columns=columns_to_remove)
X = X.drop(columns=mediapipe.columns[0],axis=1)
# X = X.astype(float)
y = mediapipe['letter'].astype(float)
y.to_excel('saved_file.xlsx', index = False)

X_train = X.head(4898) 
y_train = y.head(4898)

X_test = X.tail(240) 
y_test = y.tail(240) 

# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                 stratify=y,
#                                                 random_state=42,
#                                                 test_size=0.3)
print(X_test.shape)
print(y_test.shape)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  test_size=.3,
                                                  random_state=42)

# Normalizacja danych
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# Konwersja danych do obiektów TensorFlow
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test.values, dtype=tf.float32)
X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
y_val = tf.convert_to_tensor(y_val.values, dtype=tf.float32)

# Definicja modelu
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation='softmax')
])

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Kompilacja modelu
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trening modelu
num_epochs = 60
batch_size = 20

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    callbacks=[es_callback])

# history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)

# # Obliczenie straty treningowej i testowej
# train_loss = model.evaluate(X_train, y_train, verbose=0)
# test_loss = model.evaluate(X_test, y_test, verbose=0)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

print(test_acc)

# Save the entire model as a SavedModel.
model.save('saved_model/my_model.h5')

# print(f'Training Loss: {test_loss:.4f}')
# print(f'Test Loss: {test_loss:.4f}')