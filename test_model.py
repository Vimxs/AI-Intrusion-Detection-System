import pandas as pd

df = pd.read_csv("archive (1)/KDDTrain+.txt", header=None)

print(df.head())
print(df.shape)