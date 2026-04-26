import argparse
import os
import sys

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.common import calculate_performance


kpi_metrics = [
    "02e99bd4f6cfb33f",
    "07927a9a18fa19ae",
    "18fbb1d5a5dc099d",
    "1c35dbf57f55f5e4",
    "40e25005ff8992bd",
    "71595dd7171f4540",
    "769894baefea4e9e",
    "76f4550c43334374",
    "7c189dd36f048a6c",
    "88cf3a776ba00e7c",
    "8bef9af9a922e0b3",
    "8c892e5525f3e491",
    "9ee5879409dccef9",
    "a40b1df87e3f1c87",
    "a5bf5d65261d859a",
    "affb01ca2b4f0b45",
    "cff6d3c01e6a6bfa",
    "da403e4e3f87c9e0",
    "e0770391decc44ce",
]

# kpi_metrics = [
#     "769894baefea4e9e",
#     "76f4550c43334374",
# ]

def train_and_test(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Prepare features and labels
    X = data[['value']].values
    y = data['label'].values

    # Perform a 70-30 split manually (first 70% for training, last 30% for testing)
    split_index = int(len(data) * 0.7)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Set up the hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    # # Stratified K-Fold to ensure balanced class distribution
    # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Set up GridSearch with cross-validation and class weight balancing
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        param_grid,
        scoring='f1',
        cv=3
    )
    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_clf = grid_search.best_estimator_

    # Make predictions
    y_pred = best_clf.predict(X_test)

    # Evaluate the model using F1 score
    res_dict = calculate_performance(y_pred, y_test)
    print(f"Training and testing for {file_path}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Results: {res_dict}")
    return res_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision-tree baseline for KPI CSVs")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Directory containing <metric>.csv files for each KPI",
    )
    parser.add_argument(
        "--result-path",
        required=True,
        help="Directory where results.csv will be written",
    )
    args = parser.parse_args()

    base_path = args.dataset_path
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    with open(f"{result_path}/results.csv", "w") as f:
        f.write("metric,f1,precision,recall,\n")
        for metric in kpi_metrics:
            print(f"Training and testing for {metric}")
            file_path = f"{base_path}/{metric}.csv"
            res_dict = train_and_test(file_path)
            f.write(f"{metric},{res_dict['f1']},{res_dict['precision']},{res_dict['recall']},\n")
