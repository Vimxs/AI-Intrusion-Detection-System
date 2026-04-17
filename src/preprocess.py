from sklearn.preprocessing import LabelEncoder

def preprocess_data(train, test, columns):
    train.columns = columns
    test.columns = columns

    le = LabelEncoder()

    for col in ['protocol_type', 'service', 'flag']:
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])

    X_train = train.drop(['label', 'difficulty'], axis=1)
    y_train = train['label'].apply(lambda x: "normal" if x == "normal" else "attack")

    X_test = test.drop(['label', 'difficulty'], axis=1)
    y_test = test['label'].apply(lambda x: "normal" if x == "normal" else "attack")

    return X_train, X_test, y_train, y_test