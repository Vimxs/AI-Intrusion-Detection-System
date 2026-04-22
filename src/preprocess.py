from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(train, test, columns):

    train.columns = columns
    test.columns = columns

    le = LabelEncoder()

    for col in ['protocol_type', 'service', 'flag']:
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])

    # Binary labels
    train['label'] = train['label'].apply(lambda x: "normal" if x == "normal" else "attack")
    test['label'] = test['label'].apply(lambda x: "normal" if x == "normal" else "attack")

    X_train = train.drop(['label', 'difficulty'], axis=1)
    y_train = train['label']

    X_test = test.drop(['label', 'difficulty'], axis=1)
    y_test = test['label']

    # ======================
    # SCALE DATA (IMPORTANT FIX)
    # ======================
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test