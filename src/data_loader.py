import pandas as pd

def load_data():
    train = pd.read_csv(
        "archive (1)/KDDTrain+.txt",
        header=None,
        low_memory=False
    )

    test = pd.read_csv(
        "archive (1)/KDDTest+.txt",
        header=None,
        low_memory=False
    )

    return train, test