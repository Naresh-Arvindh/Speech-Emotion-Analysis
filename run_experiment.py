import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Import from your new local files
from test_train_split import split_my_data
from cross_validation import run_kfold
from evaluate_matrix import show_performance_metrics

def main():
    
    X = np.load("X_data.npy")
    y = np.load("y_data.npy")

    X_train, X_test, y_train, y_test = split_my_data(X, y)
    
    model = RandomForestClassifier(n_estimators=300, oob_score=True, random_state=42, n_jobs=-1)
    
    run_kfold(model, X, y)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    show_performance_metrics(y_test, predictions)

if __name__ == "__main__":
    main()
