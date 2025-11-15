import pandas as pd
import math
import time
from groq import Groq
import google.generativeai as genai

# ===============================
# 1. Setup API clients
# ===============================
# fist api key = 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
groq_client = Groq(
    api_key="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")
genai.configure(api_key="AIzaSyBB17Vw0KL6mz4xmb53YEaqCvHwiZW_wO0")


# ===============================
# 2. Helper function: classify a batch
# ===============================
def classify_batch(batch_df):
    rows_text = ""
    for idx, row in batch_df.iterrows():
        rows_text += f"Row {idx}:\nText: {row['text']}\nBias type: {row['bias_rating']}\n\n"

    prompt = f"""
    You are given rows of text data. Each row contains:
    - Extracted keywords/phrases (not full coherent text).
    - A bias type (Left, Right, or Center).

    Your task: Assign exactly one **bias subtype** for each row, based on both the text and its bias type.

    ⚠️ CRITICAL: You MUST strictly follow the type-to-subtype mapping below. Do NOT assign a subtype from a different bias type.

    **MANDATORY TYPE AND SUBTYPE CONFIGURATION:**

    If bias type is "Left", you MUST choose ONLY from these subtypes:
    - Liberal
    - Secular
    - Socialist

    If bias type is "Right", you MUST choose ONLY from these subtypes:
    - Conservative
    - Nationalist
    - Capitalist

    If bias type is "Center", you MUST use:
    - Center

    **STRICT RULES:**
    1. ALWAYS check the bias type first before assigning a subtype
    2. NEVER assign "Conservative" to a "Left" bias type
    3. NEVER assign "Liberal" to a "Right" bias type
    4. NEVER assign "Socialist" to a "Right" bias type
    5. NEVER assign "Capitalist" to a "Left" bias type
    6. NEVER assign "Nationalist" to a "Left" bias type
    7. NEVER assign "Secular" to a "Right" bias type
    8. If bias type is "Center", always assign "Center" as the subtype
    9. Pick exactly ONE subtype per row from the correct category
    10. Do NOT create new categories or subtypes
    11. Do NOT explain or justify your choice
    12. Do NOT output anything other than the required format
    13. Do NOT leave any row blank

    Output format (one line per row, no extra text):
    Row <row_index>: <subtype>

    Now classify the following rows, making sure each subtype matches its parent bias type:

    {rows_text}
    """


    # Try Groq first
    try:
        response = groq_client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    # If Groq fails → fallback to Gemini
    except Exception as e:
        # print(f"⚠ Groq failed: {e} → switching to Gemini...")
        # try:
        #     model = genai.GenerativeModel("models/gemini-2.0-flash")
        #     response = model.generate_content(prompt)
        #     return response.text.strip()
        # except Exception as e2:
        #     print(f"❌ Gemini also failed: {e2}")
        #     return ""
        print(f"⚠ Groq gpt-oss failed: {e} → switching to Groq qwen3...")
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

# ===============================
# 3. Process multiple files
# ===============================
for file_number in range(1):  # 46 → 52 inclusive
    print(f"\n===============================")
    print(f"📂 Processing file: part_{file_number}.csv")
    print(f"===============================\n")

    # Read dataset
    df = pd.read_csv(f"./rows_with_nulls.csv")
    print("Total rows:", len(df))

    # Add new column
    df["type_of_biasness"] = ""

    # Process in batches of 10
    batch_size = 10
    num_batches = math.ceil(len(df) / batch_size)

    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        batch_df = df.iloc[start:end]

        print(
            f"Processing batch {i+1}/{num_batches} ({len(batch_df)} rows)...")

        output_text = classify_batch(batch_df)

        # Parse response back into dataframe
        for line in output_text.splitlines():
            if line.strip().startswith("Row"):
                try:
                    parts = line.split(":")
                    row_index = int(parts[0].replace("Row", "").strip())
                    subtype = parts[1].strip()
                    df.at[row_index, "type_of_biasness"] = subtype
                except Exception as e:
                    print(f"⚠ Parse error on line: {line} | {e}")

        time.sleep(2)  # small pause inside batches

    # Save output file
    output_file = f"./ai_splits/ai_split_{file_number}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved file with classifications: {output_file}")

    # Wait 20 seconds before next file
    print("⏳ Waiting 10 seconds before next file...\n")
    time.sleep(10)
