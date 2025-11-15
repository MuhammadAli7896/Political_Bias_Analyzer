import pandas as pd

# Configuration
input_file = "./final_without_nulls.csv"  # Change to your input file
output_file_valid = "./final_validated.csv"
output_file_invalid = "./invalid_rows.csv"

# Read the CSV file
df = pd.read_csv(input_file)

print(f"Total rows: {len(df)}")

# Define the valid mappings based on the prompt rules
# bias_rating → allowed type_of_biasness values
valid_mappings = {
    "Left": ["Liberal", "Secular", "Socialist"],
    "Right": ["Conservative", "Nationalist", "Capitalist"],
    "Center": ["Center"]
}

# Function to check if a row has valid bias_rating ↔ type_of_biasness mapping
def is_valid_mapping(row):
    bias_rating = row['bias_rating']
    type_of_biasness = row['type_of_biasness']
    
    # Handle null or empty values
    if pd.isna(bias_rating) or pd.isna(type_of_biasness):
        return False
    
    # Convert to string and strip whitespace
    bias_rating = str(bias_rating).strip()
    type_of_biasness = str(type_of_biasness).strip()
    
    # Check if empty after stripping
    if bias_rating == '' or type_of_biasness == '':
        return False
    
    # Check if bias_rating exists in our mapping
    if bias_rating not in valid_mappings:
        return False
    
    # Check if type_of_biasness is in the valid list for this bias_rating
    return type_of_biasness in valid_mappings[bias_rating]

# Apply validation
df['is_valid'] = df.apply(is_valid_mapping, axis=1)

# Separate valid and invalid rows
valid_rows = df[df['is_valid'] == True].drop(columns=['is_valid'])
invalid_rows = df[df['is_valid'] == False].drop(columns=['is_valid'])

print(f"\n✅ Valid rows (correct bias_rating ↔ type_of_biasness): {len(valid_rows)}")
print(f"❌ Invalid rows (mismatched or empty): {len(invalid_rows)}")

# Save valid rows
if len(valid_rows) > 0:
    valid_rows.to_csv(output_file_valid, index=False)
    print(f"✅ Saved valid rows to: {output_file_valid}")
else:
    print("⚠️ No valid rows found!")

# Save invalid rows for review
if len(invalid_rows) > 0:
    invalid_rows.to_csv(output_file_invalid, index=False)
    print(f"⚠️ Saved invalid rows to: {output_file_invalid}")
    
    # Show some examples of invalid mappings
    print(f"\n📊 Sample invalid mappings:")
    for idx, row in invalid_rows.head(10).iterrows():
        print(f"  Row {idx}: bias_rating='{row['bias_rating']}' → type_of_biasness='{row['type_of_biasness']}'")
else:
    print("✅ No invalid rows found!")

# Summary statistics
print(f"\n� Summary:")
print(f"  Total rows processed: {len(df)}")
print(f"  Valid rows: {len(valid_rows)} ({len(valid_rows)/len(df)*100:.2f}%)")
print(f"  Invalid rows: {len(invalid_rows)} ({len(invalid_rows)/len(df)*100:.2f}%)")
