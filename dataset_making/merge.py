import pandas as pd
import os
import glob

# ==== CONFIGURATION ====
input_dir = "without"             # Folder containing your CSV files
output_file = "final_without_nulls.csv"  # Output merged file name

# ==== FIND ALL CSV FILES ====
csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv")))

if not csv_files:
    print(f"No CSV files found in '{input_dir}' folder.")
    exit()

# ==== READ AND CONCATENATE ====
dfs = [pd.read_csv(file) for file in csv_files]
merged_df = pd.concat(dfs, ignore_index=True)

# ==== SAVE MERGED FILE ====
merged_df.to_csv(output_file, index=False)
print(f"\n✅ Successfully merged {len(csv_files)} files into '{output_file}'.")
