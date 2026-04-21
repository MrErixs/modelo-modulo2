import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from utils_text import clean_reviews_dataframe

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

# Normalizo y limpio el text
df = clean_reviews_dataframe(df)

# Separo train y test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Hago oversampling, PERO solo en train
train_majority = train_df[train_df["label"] == 1]
train_minority = train_df[train_df["label"] == 0]

# Aumento la clase menor para igualarla un poco a la de mayor cantidad
# Aumento al 20% la minoria, si la aumento al 50/50 con la mayoria, me ariesgo a que el modelo haga memorizacion
target_ratio = 0.20
minority_target_size = int((target_ratio / (1 - target_ratio)) * len(train_majority))

train_minority_upsampled = resample(
    train_minority,
    replace=True,
    n_samples=minority_target_size,
    random_state=42
)

train_df_balance = pd.concat([train_majority, train_minority_upsampled])

# Mezclo las filas para que no queden pegadas
train_df_balance = train_df_balance.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardo los archivos pre procesados
df.to_csv("data/processed/reviews_binary_clean.csv", index = False)
# df con el oversampling
train_df_balance.to_csv("data/processed/train_oversampling.csv", index = False)
# df sin el oversampling
train_df.to_csv("data/processed/train_normal.csv", index = False)
test_df.to_csv("data/processed/test.csv", index = False)

# Revision
print("Conteo de stars:")
print(df["stars"].value_counts())

print("\nConteo de label total:")
print(df["label"].value_counts())

print("\nConteo de label en train original:")
print(train_df["label"].value_counts())

print("\nConteo de label en train balanceado:")
print(train_df_balance["label"].value_counts())

print("\nConteo de label en test:")
print(test_df["label"].value_counts())