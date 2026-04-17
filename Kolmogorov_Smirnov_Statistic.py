import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

df = pd.read_csv("synthetic_authentication_dataset.csv")

benign = df[df["Label"]=="Benign"]["Risk_Score"]
attack = df[df["Label"]=="Attack"]["Risk_Score"]

ks_stat, p_value = ks_2samp(attack,benign)

print("KS Statistic:", ks_stat)
print("P-value:", p_value)

benign_sorted = np.sort(benign)
attack_sorted = np.sort(attack)

cdf_benign = np.arange(len(benign_sorted)) / len(benign_sorted)
cdf_attack = np.arange(len(attack_sorted)) / len(attack_sorted)

plt.figure(figsize=(7,5))

plt.plot(benign_sorted, cdf_benign, label="Benign CDF")
plt.plot(attack_sorted, cdf_attack, label="Attack CDF")

plt.xlabel("Risk Score")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Risk Scores")
plt.legend()

plt.grid()

plt.savefig("figure_ks_risk_separation.png", dpi=300)
plt.close()