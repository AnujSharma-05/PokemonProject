import pandas as pd

# Define the path to your CSV file.
file_path = 'pokemon.csv'

try:
    # Use pandas to read the CSV file into a DataFrame.
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully! We are now preparing the data.")

    # --- Data Cleaning and Preparation for RAG ---

    # 1. Handle missing values.
    # Many Pokemon have only one type (like Pikachu being just Electric type), so their 'type2' column contains NaN values. Without this cleaning step, these would appear as the text "nan" in your final text chunks, which would look unprofessional.


    df['type2'].fillna('', inplace=True) # Replace NaN with an empty string for type2.

    # The 'percentage_male' column is also often empty for genderless Pokémon.
    # We'll replace NaN with a descriptive string.
    
    df['percentage_male'].fillna('genderless', inplace=True)




    # 2. Convert key columns to string format.
    # When we combine columns later, they all need to be strings.
    # We'll convert the numerical stats to strings.
    df['hp'] = df['hp'].astype(str)
    df['attack'] = df['attack'].astype(str)
    df['defense'] = df['defense'].astype(str)
    df['sp_attack'] = df['sp_attack'].astype(str)
    df['sp_defense'] = df['sp_defense'].astype(str)
    df['speed'] = df['speed'].astype(str)






    # 3. Create a combined text chunk for each Pokémon.
    # This is the core of our data preparation. We combine several relevant
    # columns into a single descriptive string. This string will be our "document" for each Pokémon that the RAG model will search through.
    # We use f-strings (formatted string literals) to easily insert the values from each row into our descriptive text.

    df['text_chunk'] = df.apply( 
        #It tells pandas, "I am about to create a new column in my DataFrame, and I want to name it 'text_chunk'." The values that will populate this new column are determined by whatever is on the right side of the equals sign. df.apply() appliues the function to each row of the csv
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
        axis=1 # This tells pandas to apply the function to each row, not each column.
    )

    # --- Display the Resulting Text Chunk ---
    # Now, let's look at the new 'text_chunk' for the first Pokémon to see if it worked.
    print("\n--- Example Text Chunk for the first Pokémon ---")
    print(df.loc[0, 'text_chunk'])

    # We can also check the shape of our new DataFrame to confirm the new column was added.
    print(f"\nDataFrame shape after adding text_chunk column: {df.shape}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please make sure the 'pokemon.csv' file is in the same directory as this script.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")




#My Question: Why not add every column in text chunk?
##Great question! When creating the 'text_chunk' for each Pokémon, we aim to include the most relevant and informative attributes that would be useful for answering questions about the Pokémon. Including every single column could lead to several issues:
#Not all columns provide useful information for understanding a Pokémon.

#Many columns contain data that is not useful for answering the most common types of questions a user might ask. Including this "noise" can dilute the important information and confuse the retrieval process.




