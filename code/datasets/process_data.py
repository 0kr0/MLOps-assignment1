import pandas as pd
from sklearn.model_selection import train_test_split
import os

def process_wine_data(raw_path, processed_dir):
    # Load
    df = pd.read_csv(raw_path, sep=';')  # Wine CSV uses ';' separator

    # Clean: Remove missing (rare), impute if any, remove outliers (IQR method)
    df = df.dropna()  # Drop any NA
    # Outlier removal for numerical columns
    for col in df.columns[:-1]:  # All except 'quality'
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))]

    # Optional: Subset to 8 features if you want (e.g., drop 'density', 'pH', 'citric acid')
    # df = df.drop(columns=['density', 'pH', 'citric acid'])

    # Split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['quality'])

    # Save
    # os.makedirs(processed_dir, exist_ok=True)
    train.to_csv(f'{processed_dir}/train.csv', index=False)
    test.to_csv(f'{processed_dir}/test.csv', index=False)
    print("Data processed and saved.")
