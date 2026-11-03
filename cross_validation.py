import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

def run_kfold(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    
    print(f"Individual Fold Accuracies: {np.array2string(scores, precision=4)}")
    print(f"Mean CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    return scores
