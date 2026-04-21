# html para convertir entidades html a caracteres normales
import html
# re para limpiar patrones
import re

def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def clean_reviews_dataframe(df):
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text", "label"])
    return df