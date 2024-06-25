import pandas as pd
import os

def extract_key(filename):
    parts = filename.split('_')
    # check first 8 parts of name are same
    return '_'.join(parts[:8])

def make_metadb(csv_path, cmeasure_dataframe, output_path):
    # Check if the CSV file exists
    if not os.path.exists(csv_path):
        print(f"No CSV found at {csv_path}. Creating a new CSV at {output_path}.")
        cmeasure_dataframe.to_csv(csv_path, index=False)
    
    # Read the CSV file
    csv_data = pd.read_csv(csv_path)
    
    # Extract the key from the 'file' column in the DataFrame for matching
    cmeasure_dataframe['key'] = cmeasure_dataframe['file'].apply(extract_key)

    # Log data to debug
    print("CSV Data 'Data' column:", csv_data['Data'].head())
    print("DataFrame 'key' column:", cmeasure_dataframe['key'].head())

    # Merge the two datasets on the extracted key
    merged_data = pd.merge(csv_data, cmeasure_dataframe, left_on='Data', right_on='key', how='inner')

    if merged_data.empty:
        print("No matching data found for merging. Check the 'Data' and 'file' columns.")

    # Save the merged data to a new CSV file
    merged_data.to_csv(output_path, index=False)
