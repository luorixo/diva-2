import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

file_path = 'meta_database_alfa_svm.csv'

data = pd.read_csv(file_path)
print(data.head())

# Extract complexity measure columns and target column
complexity_measures = data.loc[:, 'c1':'t4']

# Drop columns with any NaN or empty values
complexity_measures = complexity_measures.dropna(axis=1)

# Convert the cleaned DataFrame to a NumPy array
complexity_measures = complexity_measures.to_numpy()

# Extract the target column
target = data['Test.Clean'].to_numpy()

# Flatten the complexity measure arrays into 1D arrays
flattened_complexity_measures = complexity_measures.reshape((complexity_measures.shape[0], -1))
print(flattened_complexity_measures)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(flattened_complexity_measures, target, test_size=0.2, random_state=42)

# Train an SVM model
svm_model = SVR()
svm_model.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = svm_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')

# Save the model
joblib.dump(svm_model, 'alfa_svm_metalearner.pkl')
