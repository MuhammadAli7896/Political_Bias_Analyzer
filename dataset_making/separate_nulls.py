import pandas as pd

# Configuration
input_file = "./ai_splits_merged.csv"  # Change to your input file
output_file_with_nulls = "./rows_with_nulls.csv"
output_file_without_nulls = "./rows_without_nulls.csv"
column_to_check = "type_of_biasness"  # Change to your column name

# Read the CSV file
df = pd.read_csv(input_file)

print(f"Total rows: {len(df)}")

# Separate rows with empty values (null, NaN, or empty string) in the specified column
rows_with_nulls = df[df[column_to_check].isna() | (df[column_to_check].astype(str).str.strip() == '')]
rows_without_nulls = df[~(df[column_to_check].isna() | (df[column_to_check].astype(str).str.strip() == ''))]

print(f"Rows with empty/null in '{column_to_check}': {len(rows_with_nulls)}")
print(f"Rows without empty/null in '{column_to_check}': {len(rows_without_nulls)}")

# Save to separate files
if len(rows_with_nulls) > 0:
    rows_with_nulls.to_csv(output_file_with_nulls, index=False)
    print(f"✅ Saved rows with nulls to: {output_file_with_nulls}")
else:
    print("No rows with null values found.")

if len(rows_without_nulls) > 0:
    rows_without_nulls.to_csv(output_file_without_nulls, index=False)
    print(f"✅ Saved rows without nulls to: {output_file_without_nulls}")
else:
    print("No rows without null values found.")
