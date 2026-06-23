import pandas as pd

# ================= CONFIGURAÇÕES =================
ARQUIVO_REFERENCIA = 'breast-level_annotations_final_limpo(2).csv'
ARQUIVO_FINDING = 'finding_annotations.csv'
ARQUIVO_SAIDA = 'finding_annotations_split.csv'
# =================================================

print("A carregar os ficheiros CSV...")
df_ref = pd.read_csv(ARQUIVO_REFERENCIA)
df_find = pd.read_csv(ARQUIVO_FINDING)

print("A sincronizar as divisões (splits) pelo ID do Paciente (study_id)...")

# 1. Como um paciente pode ter 2 linhas (mama esquerda e direita) no arquivo de referência,
# pegamos apenas uma ocorrência para saber em qual split o paciente caiu.
df_ref_unique = df_ref.drop_duplicates(subset=['study_id'])

# 2. Criar um "dicionário" que mapeia cada paciente (study_id) ao seu split
mapa_splits = dict(zip(df_ref_unique['study_id'], df_ref_unique['split']))

# 3. Aplicar o mapeamento ao arquivo finding baseado no study_id do paciente
df_find['split'] = df_find['study_id'].map(mapa_splits)

# 4. Limpeza de Segurança: 
# Se a anotação pertence a um paciente que foi excluído durante a sua limpeza,
# o split vai ficar vazio (NaN). Vamos descartar essas linhas.
linhas_antes = len(df_find)
df_find = df_find.dropna(subset=['split'])
linhas_removidas = linhas_antes - len(df_find)

# 5. Salvar o novo ficheiro
df_find.to_csv(ARQUIVO_SAIDA, index=False)

print("\n" + "="*50)
print("✅ SINCRONIZAÇÃO CONCLUÍDA COM SUCESSO")
print("="*50)
print(f"Ficheiro guardado como: {ARQUIVO_SAIDA}")
print(f"Anotações descartadas (pacientes ausentes no arquivo limpo): {linhas_removidas}\n")
print("Nova distribuição de dados no finding_annotations_split.csv:")
print(df_find['split'].value_counts())
print("="*50)