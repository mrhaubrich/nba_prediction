from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


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


if __name__ == "__main__":
    # Load data
    data = load_data_from_csv()
    data = data.drop(columns=["Player", "FG%.1", "Tm"])

    # Identify missing values before handling
    identify_missing_values(data)

    # Handle missing values
    data = handle_missing_values(data)

    # Extract labels before normalization
    labels = data["Performance"].to_numpy()  # Labels as strings or categories

    # Encode labels to integers
    le = LabelEncoder()
    y_labels = le.fit_transform(labels)  # y_labels are integers starting from 0
    class_labels = le.classes_

    # Remove 'Performance' column from data
    data = data.drop(columns=["Performance"])

    # Verify that missing values have been handled
    print("\nAfter handling missing values:")
    identify_missing_values(data)

    # Normalize categorical variables
    data = normalize_position(data)

    # Print the first few rows of the processed data
    print("\nProcessed Data:")
    print(data.head())

    # Extract features
    X = data.to_numpy()  # Features

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # Avoid division by zero
    X = (X - X_mean) / X_std

    # Check for NaNs in X and y
    print(f"\nNumber of NaNs in X: {np.isnan(X).sum()}")
    print(f"Number of NaNs in y: {np.isnan(y_labels).sum()}")

    # Shuffle and split the data
    indices = np.arange(len(X))
    rng = np.random.default_rng()
    rng.shuffle(indices)
    X = X[indices]
    y_labels = np.array(y_labels)[indices]

    split_ratio = 0.75
    split_index = int(len(X) * split_ratio)
    X_train = X[:split_index]
    y_train = y_labels[:split_index]
    X_test = X[split_index:]
    y_test = y_labels[split_index:]

    # Define the network architecture
    input_size = X_train.shape[1]
    output_size = len(class_labels)  # Number of classes
    hidden_layer_size = (input_size + output_size) // 2

    # Create and train the MLPClassifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        max_iter=10000,
        learning_rate_init=0.01,
        random_state=42,
    )
    mlp.fit(X_train, y_train)

    # Evaluate the model on test data
    y_pred_labels = mlp.predict(X_test)
    y_true_labels = y_test

    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true_labels, y_pred_labels)
    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            conf_matrix,
            index=class_labels,
            columns=class_labels,
        ),
    )

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_true_labels,
            y_pred_labels,
            target_names=class_labels,
        ),
    )

    # Overall accuracy
    test_accuracy = accuracy_score(y_true_labels, y_pred_labels)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")

    # Feature importance
    # Sum the absolute weights for each input feature
    # across all neurons in the first hidden layer
    importance = np.sum(np.abs(mlp.coefs_[0]), axis=1)
    # Normalize the importance scores
    importance /= np.sum(importance)
    feature_names = data.columns
    sorted_features = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )
    print("\nTop 10 Features Contributing to the Model:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")
