import pandas as pd

import pickle

def int_to_letter(numbers):
    alphabet = 'abcdefghiklmnopqrstuvwxy'
    encoded_letters = [alphabet[int(num) - 1] for num in numbers]
    encoded_string = ''.join(encoded_letters)
    return encoded_string

def perform_processing(csv):

    loaded_model = pickle.load(open('saved_model/best_clf.pkl', 'rb'))

    # mediapipe = pd.read_csv(csv)
    mediapipe = csv

    # Wyodrębnienie nazw kolumn
    columns = mediapipe.columns.tolist()

    # Wybór kolumn do usunięcia
    columnsErase = columns[64:]

    X = mediapipe.drop(columns=columnsErase)

    # # Columns to erase
    columns_to_remove = [col for col in X.columns if col.endswith('.z')]

    # # Erase columns
    X = X.drop(columns=columns_to_remove)
    X = X.drop(columns=mediapipe.columns[0],axis=1)

    predictions = loaded_model.predict(X)

    predictions = int_to_letter(predictions)

    unique_letters = []
    for letter in predictions:
        unique_letters.append(letter)
    
    predicted_data = pd.DataFrame(
        unique_letters,
        columns=['letter']
    )

    return predicted_data