import pandas as pd

df = pd.read_csv("breast-level_annotations.csv")

resultado = pd.DataFrame({
    "Quantidade": df["breast_density"].value_counts(),
    "Percentual (%)": df["breast_density"].value_counts(normalize=True).mul(100).round(2)
})

print(resultado)