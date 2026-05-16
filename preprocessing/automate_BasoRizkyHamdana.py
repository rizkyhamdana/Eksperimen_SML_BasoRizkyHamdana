import pandas as pd
import numpy as np
import os

def preprocess_data(file_path):
    """
    Fungsi untuk melakukan preprocessing data Titanic secara otomatis.
    """
    # 1. Memuat Dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")
    
    df = pd.read_csv(file_path)
    
    # 2. Menangani Data Kosong
    # Mengisi Age yang kosong dengan median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Mengisi Embarked yang kosong dengan modus
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Menghapus kolom Cabin jika ada
    if 'Cabin' in df.columns:
        df.drop(columns=['Cabin'], inplace=True)
    
    # 3. Menghapus Data Duplikat
    df.drop_duplicates(inplace=True)
    
    # 4. Feature Selection
    # Menghapus kolom yang tidak memberikan informasi prediktif yang kuat
    cols_to_drop = ['PassengerId', 'Name', 'Ticket']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    
    # 5. Encoding Data Kategorikal
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    return df

if __name__ == "__main__":
    # Path dataset
    raw_data_path = 'dataset_raw/titanic_train.csv'
    output_dir = 'preprocessing/titanic_preprocessing'
    
    # Buat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lakukan preprocessing
    try:
        processed_df = preprocess_data(raw_data_path)
        
        # Simpan hasil preprocessing
        output_file = os.path.join(output_dir, 'titanic_cleaned.csv')
        processed_df.to_csv(output_file, index=False)
        print(f"Preprocessing selesai! Data disimpan di: {output_file}")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")


# Triggering automation test #2