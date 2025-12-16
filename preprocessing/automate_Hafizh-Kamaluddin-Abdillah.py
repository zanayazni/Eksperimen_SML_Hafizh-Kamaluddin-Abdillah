from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler


def preprocess(input_file: Path, output_file: Path):
    df = pd.read_csv(input_file)

    new_df = df.copy()
    new_df = new_df.dropna()
    new_df = new_df.drop_duplicates()

    le = LabelEncoder()
    new_df["weather"] = le.fit_transform(new_df["weather"])

    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Label mapping:", label_mapping)

    columns_to_scale = ["temp_max", "temp_min", "precipitation", "wind"]
    scaler = RobustScaler()
    new_df[columns_to_scale] = scaler.fit_transform(new_df[columns_to_scale])
    new_df = new_df.drop(columns=[ "date"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_file, index=False)

    print(f"Preprocessing selesai. File disimpan di:\n{output_file}")

#main function
if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]

    input_file = (
        project_root
        / 'seattle-weather_raw.csv'
    )

    output_file = (
        project_root
        / 'preprocessing'
        / 'seattle-weather_preprocessing.csv'
    )

    preprocess(input_file, output_file)
