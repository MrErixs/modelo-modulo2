import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Cargo los datos
train_df = pd.read_csv("data/processed/train_augmented.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Tomo el texto y las etiquetas
x_train_text = train_df["text"].astype(str).tolist()
y_train = train_df["label"].values

x_test_text = test_df["text"].astype(str).tolist()
y_test = test_df["label"].values

# Tokenizacion
# num_words es el tamaño maximo del vocabulario

vocab_size = 10000
max_length = 100
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(x_train_text)

# Convierto el texto en una secuencia numerica
x_train_seq = tokenizer.texts_to_sequences(x_train_text)
x_test_seq = tokenizer.texts_to_sequences(x_test_text)

# Balanceo de informacion
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = dict(zip(classes, weights))

# Pruebo que si esta funcionando la generacion de pesos
print("GENERACION DE PESOS", class_weight)

# Padding
x_train_pad = pad_sequences(
    x_train_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

x_test_pad = pad_sequences(
    x_test_seq,
    maxlen=max_length,
    padding="post",
    truncating="post"
)

# Creo el modelo
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=24),
    GlobalAveragePooling1D(),
    Dense(24, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compilo el modelo
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Entreno el modelo
history = model.fit(
    x_train_pad,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weight,
    verbose=1
)

# Evaluo el modelo
loss, accuracy = model.evaluate(x_test_pad, y_test, verbose=1)

print("\nResultados en test:")
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4}")

# Ahora hago la prediccion (umbral de decision)
y_pred = (model.predict(x_test_pad) > 0.5).astype("int32").flatten()

# Matriz de confusion
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
# Grafico la matriz de confisión
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negativo", "Positivo"],
    yticklabels=["Negativo", "Positivo"]
)
plt.title("Matriz de confusión")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.show()

# Guardo modelo
model.save("./models/sentiment_model_6.keras")
joblib.dump(tokenizer, "./models/tokenizer_6.pkl")
joblib.dump(max_length, "./models/max_length_6.pkl")

print("\nModelo guardado como models/sentiment_model_6.keras")
print("Tokenizer guardado como models/tokenizer_6.pkl")
print("Max length guardado como models/max_length_6.pkl")