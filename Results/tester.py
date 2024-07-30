import pickle

# Load the model
model_path = 'Results/test-28-datasets/svm_metalearner.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Example test data (replace with your actual data)
test_data = [[1.5, 2.3, 3.1, 4.0]]

# Make predictions
predictions = model.predict(test_data)

print(predictions)