import pandas as pd
from utils_text import clean_reviews_dataframe

USE_OVERSAMPLING = True

train_path = (
    "data/processed/train_oversampling.csv"
    if USE_OVERSAMPLING # Si es falso, tomara la data de train sin oversampling
    else "data/processed/train_normal.csv"
)
# Cargo el train real
train_df = pd.read_csv(train_path)

# Cargo las reseñas sinteticas
synthetic_df = pd.read_csv("data/raw/synthetic_reviews.csv")

# Creo una columna que distinga de las reseñas reales con las sinteticas
if "source" not in train_df.columns:
    train_df["source"] = "real"

# Las limpio
if not synthetic_df.empty:
    synthetic_df["source"] = "synthetic"
    synthetic_df = clean_reviews_dataframe(synthetic_df)

# Concateno las reseñas sinteticas con las reales
train_augmented = pd.concat([train_df, synthetic_df], ignore_index=True)

# Mezclo
train_augmented = train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardo
train_augmented.to_csv("data/processed/train_augmented.csv", index=False)

print("Train original:", len(train_df))
print("Sinteticas agregadas:", len(synthetic_df))
print("Train final:", len(train_augmented))
print(train_augmented["label"].value_counts())
print(train_augmented["source"].value_counts())