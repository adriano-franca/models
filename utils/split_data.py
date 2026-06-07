import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Carregar o dataset agrupado que criámos anteriormente
print("A carregar o ficheiro CSV...")
df = pd.read_csv('breast-level_annotations_grouped.csv')

# 2. Obter a lista de pacientes ÚNICOS (study_id)
# O VinDr-Mammo usa o study_id para representar a sessão do paciente
unique_patients = df['study_id'].unique()
print(f"Total de pacientes únicos encontrados: {len(unique_patients)}")

# 3. Primeira divisão: separar 80% dos PACIENTES para Treino e 20% para Reserva
# O random_state=42 garante reprodutibilidade (se correr o código outra vez, o resultado será o mesmo)
patients_train, patients_temp = train_test_split(unique_patients, test_size=0.20, random_state=42)

# 4. Segunda divisão: separar os 20% de Reserva em metades iguais (10% Validação, 10% Teste)
patients_valid, patients_test = train_test_split(patients_temp, test_size=0.50, random_state=42)

print(f"Pacientes no Treino (80%): {len(patients_train)}")
print(f"Pacientes na Validação (10%): {len(patients_valid)}")
print(f"Pacientes no Teste (10%): {len(patients_test)}")

# 5. Função para mapear o paciente para o seu respetivo conjunto
def assign_split(study_id):
    if study_id in patients_train:
        return 'training'
    elif study_id in patients_valid:
        return 'validation'
    else:
        return 'test'

# 6. Aplicar a nova divisão ao DataFrame, substituindo a coluna 'split' antiga
df['split'] = df['study_id'].apply(assign_split)

# 7. Verificar a distribuição real no dataset (pois alguns pacientes podem ter só 1 mama no dataset)
print("\nDistribuição FINAL das imagens no Dataset (Percentagem):")
distribuicao = df['split'].value_counts(normalize=True) * 100
print(distribuicao)

# 8. Guardar o novo ficheiro
novo_nome = 'breast-level_annotations_grouped_80_10_10.csv'
df.to_csv(novo_nome, index=False)
print(f"\n✅ Novo ficheiro guardado com sucesso: {novo_nome}")