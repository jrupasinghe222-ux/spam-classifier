import pandas as pd

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


print(df.head())

print(df.shape)

print(df.columns)

print(df.dtypes)

print(df.isnull().sum())

print(df["label"].value_counts())

df["message_length"] = df["message"].apply(len)

print(df["message_length"].describe())

print(df.groupby("label")["message_length"].describe())