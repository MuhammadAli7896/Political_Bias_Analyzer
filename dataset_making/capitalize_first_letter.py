import pandas as pd

# Read the CSV file
input_file = "./ai_splits/ai_split_1.csv"  # Change to your file path
output_file = "./ai_splits/ai_split_1.csv"  # Output file path

# Load the data
df = pd.read_csv(input_file)

# Specify the column name you want to capitalize
column_name = "bias_rating"  # Change to your column name

# Capitalize the first letter of each word in the column
df[column_name] = df[column_name].str.title()

# Save to a new CSV file
df.to_csv(output_file, index=False)

print(f"✅ Capitalization complete! Saved to: {output_file}")