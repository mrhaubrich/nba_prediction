from pathlib import Path

import numpy as np
import pandas as pd


def load_data_from_csv() -> pd.DataFrame:
    path = Path(__file__).parent.parent / "data" / "nba_dados.csv"
    print(f"Loading data from: {path}")
    return pd.read_csv(path)


def identify_missing_values(data: pd.DataFrame) -> None:
    nan_columns = data.columns[data.isna().any()].tolist()
    if nan_columns:
        print("Columns with NaN data:", nan_columns)
        for col in nan_columns:
            num_nans = data[col].isna().sum()
            print(f"{col}: {num_nans} missing values")
    else:
        print("No missing values found.")


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    # Fill numerical columns with the median of the column
    numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
    for col in numerical_cols:
        if data[col].isna().any():
            median_value = data[col].median()
            data[col] = data[col].fillna(median_value)
    # Fill categorical columns with the mode
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if data[col].isna().any():
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)
    return data


def normalize_position(data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data, columns=["Pos"], dtype=int, dummy_na=False)


def normalize_team(data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data, columns=["Tm"], dtype=int, dummy_na=False)


def normalize_performance(data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data, columns=["Performance"], dtype=int, dummy_na=False)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_labels: np.ndarray,
) -> np.ndarray:
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_matrix[t][p] += 1
    print(f"\nConfusion Matrix (Classes: {', '.join(class_labels)}):")
    print(pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels))
    return conf_matrix


def calculate_metrics(conf_matrix: np.ndarray, class_labels: np.ndarray) -> tuple:
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

    # Print precision, recall, and F1 score for each class
    for i, label in enumerate(class_labels):
        print(f"\nClass '{label}':")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1 Score:  {f1_score[i]:.4f}")

    return precision, recall, f1_score


def compute_feature_importance(weights: list) -> np.ndarray:
    # Sum the absolute weights for each input feature across all
    # neurons in the first hidden layer
    importance = np.sum(np.abs(weights[0]), axis=1)
    # Normalize the importance scores
    importance /= np.sum(importance)
    return importance


class MLP:
    """Multi-layer Perceptron (MLP) for classification."""

    def __init__(self, layers: list) -> None:
        """Initialize the MLP with the given layer sizes.

        layers: list of layer sizes, e.g., [input_size, hidden1_size, ..., output_size]
        """
        self.layers = layers
        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights and biases for each layer."""
        # Initialize weights and biases using He initialization for ReLU activation
        rng = np.random.default_rng()
        for i in range(1, len(self.layers)):
            input_size = self.layers[i - 1]
            output_size = self.layers[i]
            weight = rng.normal(0, np.sqrt(2 / input_size), (input_size, output_size))
            bias = np.zeros((1, output_size))
            self.weights.append(weight)
            self.biases.append(bias)

    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, z)

    def relu_derivative(self, z: np.ndarray) -> np.ndarray:
        """Apply the derivative of the ReLU activation function."""
        return (z > 0).astype(float)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> list:
        """Forward pass through the network."""
        activations = [X]
        _input = X
        for i in range(len(self.weights)):
            z = np.dot(_input, self.weights[i]) + self.biases[i]
            a = self.softmax(z) if i == len(self.weights) - 1 else self.relu(z)
            activations.append(a)
            _input = a
        return activations

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the cross-entropy loss."""
        # Cross-entropy loss
        epsilon = 1e-8  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def backward(self, activations: list, y_true: np.ndarray) -> tuple:
        """Backward pass through the network to compute gradients."""
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
                    activations[i],
                )
        return grads_w, grads_b

    def update_parameters(
        self,
        grads_w: list,
        grads_b: list,
        learning_rate: float,
    ) -> None:
        """Update weights and biases using gradient descent."""
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        activations = self.forward(X)
        return activations[-1]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        learning_rate: float,
    ) -> None:
        """Train the model using mini-batch gradient descent."""
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
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate the model on test data and return the accuracy."""
        y_pred = self.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)
        return np.mean(y_pred_labels == y_true_labels)


if __name__ == "__main__":
    # Load data
    data = load_data_from_csv()
    data = data.drop(columns=["Player", "FG%.1", "Tm"])

    # Identify missing values before handling
    identify_missing_values(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Extract class labels
    class_labels = data["Performance"].unique()

    # Verify that missing values have been handled
    print("\nAfter handling missing values:")
    identify_missing_values(data)

    # Normalize categorical variables
    data = normalize_position(data)
    data = normalize_performance(data)
    # data = normalize_team(data)  # noqa: ERA001

    # Print the first few rows of the processed data
    print("\nProcessed Data:")
    print(data.head())

    # Extract features and labels
    performance_columns = [
        col for col in data.columns if col.startswith("Performance_")
    ]
    X = data.drop(columns=performance_columns).to_numpy()  # Features
    y = data[performance_columns].to_numpy()  # Labels (one-hot encoded)

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
    X = (X - X_mean) / X_std

    # Check for NaNs in X and y
    print(f"\nNumber of NaNs in X: {np.isnan(X).sum()}")
    print(f"Number of NaNs in y: {np.isnan(y).sum()}")

    # Shuffle and split the data
    indices = np.arange(len(X))
    rng = np.random.default_rng()
    rng.shuffle(indices)
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
    hidden_layer_sizes = [(input_size + output_size) // 2]

    layers = [input_size, *hidden_layer_sizes, output_size]

    # Create and train the MLP
    mlp = MLP(layers)
    mlp.train(X_train, y_train, epochs=10000, learning_rate=0.01)

    # Evaluate the model on test data
    y_pred_probs = mlp.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Compute confusion matrix using custom labels
    num_classes = y_test.shape[1]
    conf_matrix = compute_confusion_matrix(
        y_true_labels,
        y_pred_labels,
        num_classes,
        class_labels,
    )

    # Calculate and display metrics with custom labels
    precision, recall, f1_score = calculate_metrics(conf_matrix, class_labels)

    # Overall accuracy
    test_accuracy = np.mean(y_pred_labels == y_true_labels)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Feature importance
    feature_importance = compute_feature_importance(mlp.weights)
    feature_names = data.drop(columns=performance_columns).columns
    sorted_features = sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 10 Features Contributing to the Model:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")
