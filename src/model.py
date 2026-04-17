from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def train_models(X_train, y_train):
    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()

    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    return rf, dt