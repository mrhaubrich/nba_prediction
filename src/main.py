from pathlib import Path

import numpy as np
import pandas as pd


def load_data_from_csv():
    path = Path(__file__).parent.parent / "data" / "nba_dados.csv"
    print(f"Loading data from: {path}")
    return pd.read_csv(path)


def identify_missing_values(data: pd.DataFrame):
    nan_columns = data.columns[data.isna().any()].tolist()
    if nan_columns:
        print("Columns with NaN data:", nan_columns)
        for col in nan_columns:
            num_nans = data[col].isna().sum()
            print(f"{col}: {num_nans} missing values")
    else:
        print("No missing values found.")


def handle_missing_values(data: pd.DataFrame):
    # Fill numerical columns with the median of the column
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_cols:
        if data[col].isna().any():
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)
    # Fill categorical columns with the mode
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if data[col].isna().any():
            mode_value = data[col].mode()[0]
            data[col].fillna(mode_value, inplace=True)
    return data


def normalize_position(data: pd.DataFrame):
    data = pd.get_dummies(data, columns=["Pos"], dtype=int, dummy_na=False)
    return data


def normalize_team(data: pd.DataFrame):
    data = pd.get_dummies(data, columns=["Tm"], dtype=int, dummy_na=False)
    return data


def normalize_performance(data: pd.DataFrame):
    data = pd.get_dummies(data, columns=["Performance"], dtype=int, dummy_na=False)
    return data


def compute_confusion_matrix(y_true, y_pred, num_classes):
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t][p] += 1
    return conf_matrix


def calculate_metrics(conf_matrix):
    num_classes = conf_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FP = conf_matrix[:, i].sum() - TP
        FN = conf_matrix[i, :].sum() - TP
        precision[i] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall[i] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score[i] = (
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            if (precision[i] + recall[i]) > 0
            else 0.0
        )

    return precision, recall, f1_score


def compute_feature_importance(weights):
    # Sum the absolute weights for each input feature across all neurons in the first hidden layer
    importance = np.sum(np.abs(weights[0]), axis=1)
    # Normalize the importance scores
    importance /= np.sum(importance)
    return importance


class MLP:
    def __init__(self, layers):
        """
        layers: list of layer sizes, e.g., [input_size, hidden1_size, ..., output_size]
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights and biases using He initialization for ReLU activation
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(
                2 / self.layers[i]
            )
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = [X]
        input = X
        for i in range(len(self.weights)):
            z = np.dot(input, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:
                # Output layer with softmax activation
                a = self.softmax(z)
            else:
                # Hidden layers with ReLU activation
                a = self.relu(z)
            activations.append(a)
            input = a
        return activations

    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss
        epsilon = 1e-8  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def backward(self, activations, y_true):
        grads_w = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)
        y_pred = activations[-1]
        delta = y_pred - y_true  # For softmax and cross-entropy
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            grads_w[i] = np.dot(a_prev.T, delta) / y_true.shape[0]
            grads_b[i] = np.sum(delta, axis=0, keepdims=True) / y_true.shape[0]
            if i != 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(
                    activations[i]
                )
        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def predict(self, X):
        activations = self.forward(X)
        return activations[-1]

    def train(self, X_train, y_train, epochs=1000, learning_rate=0.001):
        for epoch in range(epochs):
            # Forward propagation
            activations = self.forward(X_train)
            # Compute loss
            loss = self.compute_loss(y_train, activations[-1])
            if np.isnan(loss):
                print(f"Epoch {epoch + 1}: Loss is NaN, stopping training.")
                break
            # Backward propagation
            grads_w, grads_b = self.backward(activations, y_train)
            # Update parameters
            self.update_parameters(grads_w, grads_b, learning_rate)
            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        accuracy = np.mean(y_pred_labels == y_true_labels)
        return accuracy


if __name__ == "__main__":
    # Load data
    data = load_data_from_csv()
    data = data.drop(columns=["Player", "FG%.1", "Tm"])

    # Identify missing values before handling
    identify_missing_values(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Verify that missing values have been handled
    print("\nAfter handling missing values:")
    identify_missing_values(data)

    # Normalize categorical variables
    data = normalize_position(data)
    # data = normalize_team(data)
    data = normalize_performance(data)

    # Print the first few rows of the processed data
    print("\nProcessed Data:")
    print(data.head())

    # Extract features and labels
    performance_columns = [
        col for col in data.columns if col.startswith("Performance_")
    ]
    X = data.drop(columns=performance_columns).values  # Features
    y = data[performance_columns].values  # Labels (one-hot encoded)

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
    X = (X - X_mean) / X_std

    # Check for NaNs in X and y
    print(f"\nNumber of NaNs in X: {np.isnan(X).sum()}")
    print(f"Number of NaNs in y: {np.isnan(y).sum()}")

    # Shuffle and split the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    split_ratio = 0.75
    split_index = int(len(X) * split_ratio)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Define the network architecture
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]  # Number of classes
    hidden_layer_sizes = [128, 64, 64, 32]

    layers = [input_size] + hidden_layer_sizes + [output_size]

    # Create and train the MLP
    mlp = MLP(layers)
    mlp.train(X_train, y_train, epochs=1000, learning_rate=0.01)

    # Evaluate the model on test data
    y_pred_probs = mlp.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    num_classes = y_test.shape[1]
    conf_matrix = compute_confusion_matrix(y_true_labels, y_pred_labels, num_classes)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # Calculate metrics
    precision, recall, f1_score = calculate_metrics(conf_matrix)

    # Display metrics
    for i in range(num_classes):
        print(f"\nClass {i}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1 Score:  {f1_score[i]:.4f}")

    # Overall accuracy
    test_accuracy = np.mean(y_pred_labels == y_true_labels)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Feature importance
    feature_importance = compute_feature_importance(mlp.weights)
    feature_names = data.drop(columns=performance_columns).columns
    sorted_features = sorted(
        zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True
    )
    print("\nTop 10 Features Contributing to the Model:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")
