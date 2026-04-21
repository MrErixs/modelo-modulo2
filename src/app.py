import streamlit as st
import joblib 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cargo los recursos
model = load_model("./models/sentiment_model_6.keras")
tokenizer = joblib.load("./models/tokenizer_6.pkl")
max_len = joblib.load("./models/max_length_6.pkl")

# Funcion de preprocesado
def prepare_text(text: str):
    text = text.strip().lower()
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded

# Funcion de prediccion
def predict(text: str):
    x = prepare_text(text)
    prob = model.predict(x, verbose=0)[0][0]

    # Ajusto el umbral
    umbral = 0.5

    if prob >= umbral:
        clas = "Positive"
    else:
        clas = "Negative"
    return clas, float(prob)

# Interfaz
st.title("Recipe review sorter")
st.write("Write a recipe or review and the model will make a prediction.")

text_user = st.text_area("Input text", height=200)

if st.button("Predict"):
    if text_user.strip() == "":
        st.warning("Please, write a text")
    else:
        clas, prob = predict(text_user)

        st.subheader("Result")
        st.write(f"**Predict:** {clas}")
        st.write(f"**Positive class probability:** {prob:.4f}")

        if clas == "Positive":
            st.success("The model considers the text to be positive.")
        else:
            st.error("The model considers the text to be negative.")
        st.write("Texto limpio:", text_user.strip().lower())
        st.write("Secuencia:", tokenizer.texts_to_sequences([text_user.strip().lower()]))