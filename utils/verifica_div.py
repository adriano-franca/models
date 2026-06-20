import pandas as pd

df = pd.read_csv('breast-level_annotations_final_limpo(2).csv')

total_pacientes = df['study_id'].nunique()

print(f"Total de pacientes (exames) únicos no dataset: {total_pacientes}")
print("-" * 50)

if 'split' in df.columns:
    print("Distribuição de pacientes por conjunto:")
    distribuicao = df.groupby('split')['study_id'].nunique().reset_index()
    distribuicao.columns = ['Conjunto', 'Quantidade de Pacientes']
    print(distribuicao.to_string(index=False))