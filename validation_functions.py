from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import make_scorer,get_scorer

def repeated_train_test(model, X, y, n, scoring):

    scorer = get_scorer(scoring)
    scores = []
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size= 0.1 + np.random.rand()*0.2)

        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        
        scores.append(scorer._score_func(y_test,y_pred))

    return scores





    