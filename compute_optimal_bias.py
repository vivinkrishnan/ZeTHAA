import pandas as pd

df = pd.read_csv("synthetic_authentication_dataset.csv")

benign = df[df["Label"]=="Benign"]["Risk_Score"]
attack = df[df["Label"]=="Attack"]["Risk_Score"]

mean_benign = benign.mean()
mean_attack = attack.mean()

delta = mean_attack - mean_benign

bias = delta / 2

print("Mean Benign", mean_benign)
print("Mean Attack", mean_attack)
print("Delta", delta)
print("Recommended Bias", bias)

threshold = df["Risk_Score"].quantile(0.5)
subset = df[df["Risk_Score"] < threshold]

delta = (subset[subset["Label"]=="Attack"]["Risk_Score"].mean() - subset[subset["Label"]=="Benign"]["Risk_Score"].mean())
print("Delta for Subset", delta)
print("Bias for Subset", delta/2)