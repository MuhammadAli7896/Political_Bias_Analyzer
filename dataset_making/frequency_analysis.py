import pandas as pd

# Configuration
input_file = "./final_validated.csv"  # Change to your input file

# Read the CSV file
df = pd.read_csv(input_file)

print(f"Total rows: {len(df)}\n")
print("=" * 60)

# ====== 1. Frequency of bias_rating ======
print("\n📊 FREQUENCY OF bias_rating:")
print("=" * 60)
bias_rating_freq = df['bias_rating'].value_counts()
print(bias_rating_freq)
print("\nPercentages:")
print(bias_rating_freq / len(df) * 100)

# ====== 2. Frequency of type_of_biasness ======
print("\n" + "=" * 60)
print("\n📊 FREQUENCY OF type_of_biasness:")
print("=" * 60)
type_of_biasness_freq = df['type_of_biasness'].value_counts()
print(type_of_biasness_freq)
print("\nPercentages:")
print(type_of_biasness_freq / len(df) * 100)

# ====== 3. Cross-tabulation (bias_rating vs type_of_biasness) ======
print("\n" + "=" * 60)
print("\n📊 CROSS-TABULATION (bias_rating vs type_of_biasness):")
print("=" * 60)
cross_tab = pd.crosstab(df['bias_rating'], df['type_of_biasness'], margins=True)
print(cross_tab)

# ====== 4. Summary Statistics ======
print("\n" + "=" * 60)
print("\n📈 SUMMARY STATISTICS:")
print("=" * 60)
print(f"\nUnique bias_rating values: {df['bias_rating'].nunique()}")
print(f"Unique type_of_biasness values: {df['type_of_biasness'].nunique()}")

print("\nBias Rating Distribution:")
for rating, count in bias_rating_freq.items():
    percentage = (count / len(df)) * 100
    print(f"  {rating}: {count} ({percentage:.2f}%)")

print("\nType of Biasness Distribution:")
for biasness, count in type_of_biasness_freq.items():
    percentage = (count / len(df)) * 100
    print(f"  {biasness}: {count} ({percentage:.2f}%)")

# ====== 5. Save frequency tables to CSV ======
# output_file_bias_rating = "./frequency_bias_rating.csv"
# output_file_type_biasness = "./frequency_type_of_biasness.csv"
# output_file_crosstab = "./frequency_crosstab.csv"

# bias_rating_freq.to_csv(output_file_bias_rating, header=['Count'])
# type_of_biasness_freq.to_csv(output_file_type_biasness, header=['Count'])
# cross_tab.to_csv(output_file_crosstab)

# print(f"\n✅ Frequency tables saved:")
# print(f"   - {output_file_bias_rating}")
# print(f"   - {output_file_type_biasness}")
# print(f"   - {output_file_crosstab}")
