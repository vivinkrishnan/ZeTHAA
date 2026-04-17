import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("synthetic_authentication_dataset.csv")

benign=df[df["Label"]=="Benign"]
attack=df[df["Label"]=="Attack"]

with open("risk_policy_thresholds.txt") as f:
    tau1,tau2=map(float,f.read().split(","))
    
attacks = df[df["Label"]=="Attack"]
stealth_attacks = attacks[attacks["Risk_Score"] < tau1]
stealth_rate = len(stealth_attacks) / len(attacks)

print("Total Attacks:", len(attacks))
print("Stealth Attacks:", len(stealth_attacks))
print("Stealth Attack Rate:", stealth_rate)

plt.figure(figsize=(7,5))

plt.scatter(benign["Risk_Score"],benign["Trust_Score"],alpha=0.2, label="Benign")
plt.scatter(attack["Risk_Score"],attack["Trust_Score"],alpha=0.2, label="Attack")

plt.axvline(tau1,color = 'orange', linestyle='--', label="Step-Up Threshold")
plt.axvline(tau2,color = 'red', linestyle='--', label="Block Threshold")

plt.axvspan(0, tau1, color='green', alpha=0.08)
plt.text(0.03,0.70, "Stealth Attack Region\n(Low-signal attacks)", fontsize=9)

plt.xlabel("Risk Score")
plt.ylabel("Trust Score")
plt.title("Risk–Trust Phase Diagram")
plt.legend()
plt.grid()

plt.savefig("risk_trust_phase_improved.png", dpi=300)
# plt.show()

plt.figure(figsize=(7,5))

plt.hist(attacks["Risk_Score"], bins=50, alpha=0.7)
plt.axvline(tau1, linestyle="--")

plt.title("Attack Risk Distribution with Stealth Region")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.grid()

plt.savefig("stealth_attack_risk_distribution_improved.png", dpi=300)
# plt.show()