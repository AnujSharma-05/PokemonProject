import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Load the local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

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

    print("\nCreating batch embeddings for all Pokémon using SentenceTransformers...")
    print(f"Total Pokémon to embed: {len(all_text_chunks)}")

    # 9. Generate embeddings locally
    embeddings = model.encode(all_text_chunks, show_progress_bar=True)

    # 10. Add the new 'embeddings' column back to the DataFrame.
    if len(embeddings) == len(df):
        df['embedding'] = list(embeddings)
        print("\nBatch embeddings created and added to the DataFrame successfully!")
        print(f"Storing embeddings in chromaDB...")
        client = chromadb.PersistentClient(path="pokemon_db")
        try:
            collection = client.create_collection(name="pokemon_embeddings")
        except:
            collection = client.get_collection(name="pokemon_embeddings")

        ids = [str(i) for i in range(len(df))]
        embeddings_list = df['embedding'].tolist()
        documents = df['text_chunk'].tolist()
        metadatas = [{"name": row['name'], "type1": row['type1'], "type2": row['type2']} for _, row in df.iterrows()]

        collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )   
        

        print(f"Successfully stored {len(df)} Pokémon embeddings in ChromaDB!")
        
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