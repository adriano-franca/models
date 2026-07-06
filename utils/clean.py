import pandas as pd

# 1. Carrega o seu CSV final
df = pd.read_csv('breast-level_annotations_grouped_80_10_10(2).csv')

def limpar_caminho(caminho):
    # Pega sempre nas duas últimas partes do caminho (a pasta do paciente e o ficheiro)
    partes = caminho.split('/')
    caminho_limpo = f"{partes[-2]}/{partes[-1]}"
    return caminho_limpo

# 2. Aplica a limpeza às duas colunas
df['path_cc'] = df['path_cc'].apply(limpar_caminho)
df['path_mlo'] = df['path_mlo'].apply(limpar_caminho)

# 3. Guarda o novo ficheiro "perfeito"
novo_nome = 'breast-level_annotations_final_limpo(2).csv'
df.to_csv(novo_nome, index=False)
print(f"✅ Limpeza concluída! Ficheiro guardado como: {novo_nome}")

# Mostra como ficou a primeira linha para confirmar
print("\nExemplo de como ficou:")
print(df[['path_cc', 'path_mlo']].head(1).values)