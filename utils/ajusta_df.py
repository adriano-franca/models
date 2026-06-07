import pandas as pd

# 1. Lê o arquivo original
df = pd.read_csv('breast-level_annotations_target.csv')

# 2. Cria a coluna com o caminho final do arquivo DICOM
# Ajuste a extensão ou pasta base conforme onde você salvou as imagens no seu PC
df['file_path'] = '/images/' + df['study_id'] + '/' + df['image_id'] + '.dicom'

# 3. Mantém apenas as colunas úteis
df_subset = df[['study_id', 'laterality', 'view_position', 'file_path', 'target', 'split']]

# 4. TRATAMENTO CRÍTICO: Remove imagens duplicadas da mesma incidência (mantém a 1ª foto tirada)
df_subset = df_subset.drop_duplicates(subset=['study_id', 'laterality', 'view_position'], keep='first')

# 5. Pivota a tabela: CC e MLO viram colunas
df_pivoted = df_subset.pivot(
    index=['study_id', 'laterality', 'target', 'split'], 
    columns='view_position', 
    values='file_path'
).reset_index()

# 6. Renomeia e limpa a tabela
df_pivoted = df_pivoted.rename(columns={'CC': 'path_cc', 'MLO': 'path_mlo'})
df_pivoted.columns.name = None

# 7. Remove pacientes que não possuem as DUAS vistas da mama (falta a CC ou falta a MLO)
df_final = df_pivoted.dropna(subset=['path_cc', 'path_mlo'])

# 8. Salva o novo DataFrame para ser usado no treinamento
df_final.to_csv('breast-level_annotations_grouped.csv', index=False)