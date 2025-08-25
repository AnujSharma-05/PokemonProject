import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file.
# This securely loads your API key.
load_dotenv()

# Configure the Gemini API with your API key.
# The `os.getenv` function retrieves the value of the environment variable.
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    exit()

# Define the path to your CSV file.
file_path = 'pokemon.csv'

try:
    # Load the data from the CSV file into a DataFrame.
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully! Preparing data for embeddings.")

    # --- Data Cleaning and Preparation for RAG ---
    df['type2'].fillna('', inplace=True)
    df['percentage_male'].fillna('genderless', inplace=True)
    df['hp'] = df['hp'].astype(str)
    df['attack'] = df['attack'].astype(str)
    df['defense'] = df['defense'].astype(str)
    df['sp_attack'] = df['sp_attack'].astype(str)
    df['sp_defense'] = df['sp_defense'].astype(str)
    df['speed'] = df['speed'].astype(str)

    # Create a combined text chunk for each Pokémon.
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

    print("Data preparation complete! Proceeding to create embeddings.")
    # --- Create Embeddings and Store Them ---

    # Define the embedding model to use.
    embedding_model = 'models/embedding-001' # actual google model which convert the text into vector embeddings

    # Create a list to store our structured data.
    # Each item will be a dictionary with the text chunk and its embedding.
    pokemon_data = []

    print("\nCreating embeddings for each Pokémon... This might take a moment.")
    for index, row in df.iterrows():
        try:
            # Generate the embedding for the text chunk.
            # genai.embed_content(...): This is the function call to the Google AI library. You are telling it, "Take the text I'm giving you and convert it into an embedding vector."
            embedding = genai.embed_content(
                model=embedding_model,
                content=row['text_chunk']
            )['embedding']
            
            # Store the original text and the new embedding.
            pokemon_data.append({
                'name': row['name'],
                'text_chunk': row['text_chunk'],
                'embedding': embedding
            })
        except Exception as embed_e:
            print(f"Error creating embedding for {row['name']}: {embed_e}")
            continue # Continue to the next Pokémon if there's an error.

    print(f"\nSuccessfully created embeddings for {len(pokemon_data)} Pokémon.")
    
    # Let's print the first Pokémon's data to see what the embedding looks like.
    print("\n--- Example of the first Pokémon's stored data ---")   
    first_pokemon = pokemon_data[0]
    print(f"Pokémon Name: {first_pokemon['name']}")
    print(f"Text Chunk: {first_pokemon['text_chunk'][:100]}...") # Print a truncated version
    print(f"Embedding Vector (first 5 values): {first_pokemon['embedding'][:5]}...") # Print a truncated version

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the 'pokemon.csv' file is in the same directory as this script.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
