import pandas as pd

df = pd.read_csv('../breast-level_annotations.csv')

df['target'] = df['breast_birads'].map({
    'BI-RADS 1': 0,
    'BI-RADS 2': 0,
    'BI-RADS 3': 1,
    'BI-RADS 4': 1,
    'BI-RADS 5': 1
})

df.to_csv('../breast-level_annotations_target.csv', index=False)

print(df[['breast_birads', 'target']].head())