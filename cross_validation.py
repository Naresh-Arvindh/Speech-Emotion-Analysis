import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

def run_kfold(model, X, y, groups):
    logo = LeaveOneGroupOut()
    
    scores = cross_val_score(model, X, y, groups=groups, cv=logo, scoring='accuracy')
    
    print(f"Mean Speaker-Independent CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores
