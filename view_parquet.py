import pandas as pd
import os

parquet_files_folder = '../persons-photo-concept-bucket-images-to-train/open_pose_controlnet/data'
file_list = os.listdir(parquet_files_folder)
file_list.sort(reverse=True)

for file_name in file_list[:10]:
    if file_name.endswith('.parquet'):
        file_path = os.path.join(parquet_files_folder, file_name)

        df = pd.read_parquet(file_path)
        
        print(f"File: {file_name}")
        print(f"Number of rows: {len(df)}")
        print(f"Columns: {df.columns}")

        print(df.head(5))

        print("First row:")
        print(df.iloc[0])

        print("\n\n")
        