import numpy as np

def split_my_data(X, y, actors):
    test_speakers = [21, 22, 23, 24]
    

    test_mask = np.isin(actors, test_speakers)
    train_mask = ~test_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_actors = actors[train_mask]
    
    print(f"Speakers assigned to Training: {np.unique(train_actors)}")
    print(f"Speakers assigned to Testing:  {np.unique(actors[test_mask])}")
    
    return X_train, X_test, y_train, y_test, train_actors
