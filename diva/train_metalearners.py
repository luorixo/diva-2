import os
import itertools
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

# Folder containing all meta database CSVs
folder_path = 'metadbs'

# Get all the CSV files from the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Function to load a meta database CSV
def load_meta_database(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to train and evaluate a model on given features and targets
def train_and_evaluate_svm(X, y, model_name):
    # Flatten the complexity measure arrays into 1D arrays
    X_flattened = X.reshape((X.shape[0], -1))
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)
    
    # Train an SVM model
    svm_model = SVR()
    svm_model.fit(X_train, y_train)
    
    # Predict and evaluate the model
    y_pred = svm_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f'Mean Squared Error for {model_name}: {mse}')
    
    # Save the model
    joblib.dump(svm_model, f'{model_name}.pkl')

# Function to combine multiple meta databases by concatenating rows
def combine_meta_databases(file_paths):
    combined_data = pd.DataFrame()
    
    for file_path in file_paths:
        data = load_meta_database(file_path)
        combined_data = pd.concat([combined_data, data], axis=0)  # Concatenate rows
    
    return combined_data

# Iterate through all possible combinations of the meta databases
for r in range(1, len(csv_files) + 1):
    for combination in itertools.combinations(csv_files, r):
        # Full paths to the meta database files
        file_paths = [os.path.join(folder_path, file) for file in combination]
        
        # Load and combine the meta databases by adding rows
        combined_data = combine_meta_databases(file_paths)
        
        # Extract complexity measure columns and target column
        complexity_measures = combined_data.loc[:, 'c1':'t4'].dropna(axis=1).to_numpy()
        target = combined_data['Test.Clean'].to_numpy()

        # Create a unique model name based on the combination of meta database files
        model_name = '_'.join([os.path.splitext(f)[0] for f in combination])
        
        # Train and evaluate the SVM model
        train_and_evaluate_svm(complexity_measures, target, model_name)
