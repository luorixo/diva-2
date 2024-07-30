import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_model(file_path):
    try:
        model = joblib.load(file_path)
        return model
    except (FileNotFoundError, IOError, joblib.externals.loky.process_executor._RemoteTraceback) as e:
        print(f"Error loading model file: {e}")
        return None

def load_test_data(metadata_path):
    metadata_df = pd.read_csv(metadata_path)
    return metadata_df

def main():
    # Test data path
    testdata_path = 'results/synth/metadata.csv'
    model_path = './svm_metalearner.pkl'  # Path to the model

    # Load model
    model = load_model(model_path)
    if model is None:
        print("Failed to load the model. Exiting...")
        return

    # Load test data
    metadata_df = load_test_data(testdata_path)

    # Extract complexity measure columns
    complexity_columns = metadata_df.loc[:, 'c1':'t4']
    complexity_columns = complexity_columns.dropna(axis=1)  # Drop columns with any NaN or empty values

    # Iterate over each row in the metadata CSV
    for index, row in metadata_df.iterrows():
        test_label = row['Test.Clean']
        
        # Extract and process complexity measures for the current row
        complexity_measures = row[complexity_columns.columns].to_numpy().reshape(1, -1)  # Flatten and reshape to 1D array
        
        try:
            # Make a prediction
            y_pred = model.predict(complexity_measures)
            
            # Calculate and print the difference
            difference = test_label - y_pred[0]
            print(f"Row {index}: Predicted = {y_pred[0]}, Actual = {test_label}, Difference = {difference}")
        
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

if __name__ == "__main__":
    main()