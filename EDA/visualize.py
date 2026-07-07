import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(
    "data/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

df["message_length"] = df["message"].apply(len)

plt.hist(
    df[df["label"] == "ham"]["message_length"],
    bins=30,
    alpha=0.7,
    label="Ham"
)

plt.hist(
    df[df["label"] == "spam"]["message_length"],
    bins=30,
    alpha=0.7,
    label="Spam"
)

plt.xlabel("Message Length")
plt.ylabel("Number of Messages")
plt.title("Distribution of Message Lengths")
plt.legend()

plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    df[df["label"] == "ham"]["message_length"],
    bins=30,
    alpha=0.5,
    density=True,
    label="Ham"
)

plt.hist(
    df[df["label"] == "spam"]["message_length"],
    bins=30,
    alpha=0.5,
    density=True,
    label="Spam"
)

plt.xlabel("Message Length")
plt.ylabel("Density")
plt.title("Distribution of Message Lengths")
plt.legend()

plt.show()