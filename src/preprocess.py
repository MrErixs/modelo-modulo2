import pandas as pd
from sklearn.model_selection import train_test_split
# html para convertir entidades html a caracteres normales
import html
# re para limpiar patrones
import re

#  Cargo el dataset
df = pd.read_csv("data/raw/Recipe Reviews and User Feedback Dataset.csv")

# Tomo "text" y "stars"
df = df[["text", "stars"]].copy()

# Elimino los nulos
df = df.dropna(subset=["text", "stars"])

# Conservo solo el 1, 2, 4, 5
df = df[df["stars"].isin([1, 2, 4, 5])].copy()

# Creo la etiqueta
df["label"] = df["stars"].apply(lambda x: 0 if x in [1, 2] else 1)

# Limpio el texto
def clean_text(text):
    text = str(text)
    text = html.unescape(text) # convierte los "&#39;" a "'"
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

df["text"] = df["text"].apply(clean_text)

# Elimino el texto vacio y a los duplicados
df = df[df["text"].str.len() > 0]
df = df.drop_duplicates(subset = ["text", "label"])

# Separo train y test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Guardo los archivos pre procesados
df.to_csv("data/processed/reviews_binary_clean.csv", index = False)
train_df.to_csv("data/processed/train.csv", index = False)
test_df.to_csv("data/processed/test.csv", index = False)

# Revision de conteos, para saber la cantidad cuantos comentarios tienen x estrellas
print("Conteo de stars:")
print(df["stars"].value_counts())

print("\nConteo de label:")
print(df["label"].value_counts())

print("\nProporción de label:")
print(df["label"].value_counts(normalize=True))