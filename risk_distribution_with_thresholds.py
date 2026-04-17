import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("synthetic_authentication_dataset.csv")

benign = df[df["Label"]=="Benign"]["Risk_Score"]
attack = df[df["Label"]=="Attack"]["Risk_Score"]

with open("risk_policy_thresholds.txt") as f:
    tau1, tau2 = map(float, f.read().split(","))
    
plt.figure(figsize=(8,5))

plt.hist(attack, bins=40, color="red" ,alpha = 0.6, label="Attack",density=True)
plt.hist(benign, bins=40, color="blue",alpha = 0.6, label="Benign", density=True)

plt.axvline(tau1,color = 'orange', linestyle='--', linewidth = 2, label="Step-Up Threshold")
plt.axvline(tau2,color = 'red', linestyle='--', linewidth = 2, label="Block Threshold")

plt.xlabel("Risk Score")
plt.ylabel("Probability Density")
plt.title("Risk Score Distribution and Decision Thresholds")
plt.legend()
plt.grid()

plt.savefig("figure_risk_score_distribution_with_thresholds.png", dpi=300)
plt.show()