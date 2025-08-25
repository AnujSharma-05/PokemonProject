import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

# 1. Load the environment variables from the .env file.
# This securely loads your API key.
load_dotenv()

# 2. Configure the Gemini API with your API key.
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    # We exit the script here because we cannot proceed without the API key.
    exit()

# 3. Define the path to your CSV file.
file_path = 'pokemon.csv'

try:
    # 4. Load the data from the CSV file into a DataFrame.
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully! We are now preparing the data.")

    # --- Data Cleaning and Preparation for RAG ---
    # 5. Handle missing values.
    # We fill NaN values in 'type2' and 'percentage_male' to avoid issues
    # when creating the descriptive text chunks.
    df['type2'].fillna('', inplace=True)
    df['percentage_male'].fillna('genderless', inplace=True)

    # 6. Convert key numerical columns to string format.
    # This is necessary because we are combining them with text later.
    df['hp'] = df['hp'].astype(str)
    df['attack'] = df['attack'].astype(str)
    df['defense'] = df['defense'].astype(str)
    df['sp_attack'] = df['sp_attack'].astype(str)
    df['sp_defense'] = df['sp_defense'].astype(str)
    df['speed'] = df['speed'].astype(str)

    # 7. Create a combined text chunk for each Pokémon.
    # We use .apply(..., axis=1) to iterate through each row and create a
    # rich, descriptive string for that Pokémon. This will be our "document" for RAG.
    df['text_chunk'] = df.apply(
        lambda row: (
            f"The Pokémon {row['name']} is a {row['type1']}"
            f"{' and ' + row['type2'] if row['type2'] else ''} type."
            f" It is a {row['classfication']} and was introduced in Generation {row['generation']}."
            f" It has the following base stats: HP of {row['hp']}, Attack of {row['attack']}, Defense of {row['defense']}, "
            f"Special Attack of {row['sp_attack']}, Special Defense of {row['sp_defense']}, and Speed of {row['speed']}."
            f" It's abilities include: {row['abilities']}."
            f" Its height is {row['height_m']} meters and its weight is {row['weight_kg']} kilograms."
            f" This Pokémon has a male gender percentage of {row['percentage_male']}."
        ),
        axis=1
    )

    # --- Implement Batch Embeddings ---

    # 8. Create a list of all text chunks.
    # This is the "batch" of data we will send in one API call.
    all_text_chunks = df['text_chunk'].tolist()

    print("\nCreating batch embeddings for all Pokémon...")
    print(f"Total Pokémon to embed: {len(all_text_chunks)}")

    # 9. Make a single API call to get all embeddings at once.
    # The 'genai.embed_content' function handles batching automatically when given a list.
    response = genai.embed_content(
        model='models/embedding-001',
        content=all_text_chunks,
    )
    
    # 10. Extract the embeddings from the response.
    embeddings = response['embedding']

    # 11. Add the new 'embeddings' column back to the DataFrame.
    # We must ensure the number of embeddings matches the number of rows.
    if len(embeddings) == len(df):
        df['embedding'] = embeddings
        print("\nBatch embeddings created and added to the DataFrame successfully!")
    else:
        print("\nError: The number of embeddings does not match the number of rows.")

    print(f"\n testing the text chunk and embedding = {df.loc[55, 'text_chunk']}") 
    # 12. Display the result for the first Pokémon to confirm.
    print("\n--- First Pokémon Data with its new Embedding ---")
    print(f"Pokémon Name: {df.loc[0, 'name']}")
    print(f"Text Chunk: {df.loc[0, 'text_chunk'][:100]}...")
    print(f"Embedding Vector (first 5 values): {df.loc[0, 'embedding'][:5]}...")
    print(f"\nTotal number of columns now: {len(df.columns)}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the 'pokemon.csv' file is in the same directory as this script.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")