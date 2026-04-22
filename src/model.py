from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def train_models(X_train, y_train):

    rf = RandomForestClassifier()
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    lr = LogisticRegression(max_iter=2000)
    gb = GradientBoostingClassifier()

    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    return rf, dt, knn, lr, gb