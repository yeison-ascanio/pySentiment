import numpy as np
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# Supongamos que tienes tus datos en una lista de tuplas (texto, sentimiento)
data = [("Este es un buen día", "positivo"),
        ("Odio este horrible clima", "negativo"),
        ("La película fue excelente", "positivo"),
        ("No me gustó el servicio", "negativo")]

# Preprocesamiento de datos
stop_words = set(stopwords.words('spanish'))
ps = PorterStemmer()

def preprocess_text(text):
    # Tokenización y eliminación de signos de puntuación
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum()]
    # Eliminación de stopwords y stemming
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return tokens

# Construcción de la matriz de transición
def build_transition_matrix(data):
    # Crear un diccionario para contar las transiciones de estado
    transitions = defaultdict(lambda: defaultdict(int))

    for text, sentiment in data:
        tokens = preprocess_text(text)
        for i in range(len(tokens) - 1):
            transitions[tokens[i]][tokens[i+1]] += 1

    # Convertir el diccionario a una matriz de transición
    unique_words = list(set(word for text, _ in data for word in preprocess_text(text)))
    transition_matrix = np.zeros((len(unique_words), len(unique_words)))

    for i, word1 in enumerate(unique_words):
        for j, word2 in enumerate(unique_words):
            transition_matrix[i, j] = transitions[word1][word2]

    # Normalizar las filas para obtener probabilidades de transición
    row_sums = transition_matrix.sum(axis=1)
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    return transition_matrix, unique_words

# Entrenamiento del modelo
transition_matrix, unique_words = build_transition_matrix(data)

# Función para predecir sentimientos
def predict_sentiment(text):
    tokens = preprocess_text(text)
    state = tokens[0]
    sentiment = []
    for token in tokens[1:]:
        if state in unique_words:
            index = unique_words.index(state)
            next_state_probs = transition_matrix[index]
            next_state_index = np.argmax(next_state_probs)
            sentiment.append(unique_words[next_state_index])
            state = unique_words[next_state_index]
    return sentiment

# Ejemplo de uso
new_text = "Me encanta este libro"
predicted_sentiment = predict_sentiment(new_text)
print(predicted_sentiment)
