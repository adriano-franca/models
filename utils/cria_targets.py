import pandas as pd

df = pd.read_csv('./breast-level_annotations.csv')

# Atualização: BI-RADS 1, 2 e 3 agora são considerados Saudáveis (0)
df['target'] = df['breast_birads'].map({
    'BI-RADS 1': 0,
    'BI-RADS 2': 0,
    'BI-RADS 3': 0, 
    'BI-RADS 4': 1,
    'BI-RADS 5': 1
})

df.to_csv('./breast-level_annotations_target_2.csv', index=False)

print(df[['breast_birads', 'target']].value_counts()) # Mudei para value_counts para você ver a nova distribuição