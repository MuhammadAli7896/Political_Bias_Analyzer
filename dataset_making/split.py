import pandas as pd
import math
import os

# ==== CONFIGURATION ====
input_file = "cleaned_balanced.csv"       # Path to your main CSV file
output_dir = "splits"          # Folder where split files will be saved
rows_per_file = 250            # Number of rows per split file

# ==== LOAD CSV ====
df = pd.read_csv(input_file)
total_rows = len(df)

# ==== CREATE OUTPUT FOLDER ====
os.makedirs(output_dir, exist_ok=True)

# ==== SPLIT AND SAVE ====
num_files = math.ceil(total_rows / rows_per_file)

for i in range(num_files):
    start = i * rows_per_file
    end = start + rows_per_file
    split_df = df.iloc[start:end]

    output_path = os.path.join(output_dir, f"data_part_{i+1}.csv")
    split_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(split_df)} rows)")

print(
    f"\n✅ Split complete! {num_files} files created in '{output_dir}' folder.")
