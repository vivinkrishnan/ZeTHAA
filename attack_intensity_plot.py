import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("attack_intensity_results.csv")

plt.figure()

plt.plot(df["Attack_Ratio"], df["Accuracy"], marker='o', label="Accuracy")
plt.plot(df["Attack_Ratio"], df["Precision"], marker='o', label="Precision")
plt.plot(df["Attack_Ratio"], df["Recall"], marker='o', label="Recall")
plt.plot(df["Attack_Ratio"], df["F1"], marker='o', label="F1")
plt.plot(df["Attack_Ratio"], df["AUC"], marker='o', label="AUC")

plt.xlabel("Attack Ratio")
plt.ylabel("Score")

plt.title("Framework Robustness under Increasing Attack Intensity")

plt.legend(title="Performance Metrics", loc="lower right")

plt.savefig("attack_intensity_robustness.png")