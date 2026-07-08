import pandas as pd

# Carrega os arquivos
finding = pd.read_csv("finding_annotations.csv")
finding_split = pd.read_csv("finding_annotations_split.csv")

# Remove a coluna breast_density caso ela já exista
if "breast_density" in finding_split.columns:
    finding_split = finding_split.drop(columns=["breast_density"])

# Cria o mapeamento study_id -> breast_density
density_map = (
    finding[["study_id", "breast_density"]]
    .drop_duplicates(subset="study_id")
)

# Adiciona a densidade ao arquivo split
finding_split = finding_split.merge(
    density_map,
    on="study_id",
    how="left"
)

# Reorganiza as colunas na ordem desejada
finding_split = finding_split[
    [
        "study_id",
        "series_id",
        "image_id",
        "laterality",
        "view_position",
        "height",
        "width",
        "breast_birads",
        "breast_density",
        "finding_categories",
        "finding_birads",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "split",
    ]
]

# Salva o resultado
finding_split.to_csv("finding_annotations_split_density.csv", index=False)

print("Arquivo salvo com sucesso!")