import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import numpy as np

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------

df = pd.read_csv("synthetic_authentication_dataset.csv")

df["Attack_Flag"] = (df["Label"]=="Attack").astype(int)

# -----------------------------------------------------
# FIGURE 1 – Attack Distribution
# -----------------------------------------------------

plt.figure()
df["Attack_Type"].value_counts().plot(kind="bar")
plt.title("Attack Type Distribution")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig1_attack_distribution.png")


# -----------------------------------------------------
# FIGURE 2 – Risk Score Distribution
# -----------------------------------------------------

plt.figure()
sns.histplot(df["Risk_Score"], bins=30, kde=True)
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("fig2_risk_distribution.png")


# -----------------------------------------------------
# FIGURE 3 – Benign vs Attack Risk
# -----------------------------------------------------

plt.figure()
sns.histplot(df[df["Label"]=="Benign"]["Risk_Score"], bins=30, color="blue", label="Benign", alpha=0.5)
sns.histplot(df[df["Label"]=="Attack"]["Risk_Score"], bins=30, color="red", label="Attack", alpha=0.5)
plt.legend()
plt.title("Risk Score by Class")
plt.xlabel("Risk Score")
plt.tight_layout()
plt.savefig("fig3_risk_by_class.png")

#-----------------------------------------------------
# Weekday, Weekend distribution
#-----------------------------------------------------

plt.figure()
login_dist = df.groupby(["Weekday","Label"]).size().unstack()
login_dist.plot(kind="bar", stacked=True)
plt.title("Login Distribution by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Count")
plt.savefig("login_weekday_distribution.png")

# -----------------------------------------------------
# FIGURE 4 – ROC Curve
# -----------------------------------------------------

labels = df["Attack_Flag"]
risk = df["Risk_Score"]

fpr, tpr, thr = roc_curve(labels, risk)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_roc_curve.png")


# -----------------------------------------------------
# FIGURE 5 – Feature Importance
# -----------------------------------------------------

weights = pd.read_csv("calibrated_weights.csv")

plt.figure()
weights.sort_values("Weight").plot(
    x="Feature",
    y="Weight",
    kind="barh",
    legend=False
)
plt.title("Contextual Feature Importance")
plt.xlabel("Weight")
plt.tight_layout()
plt.savefig("fig5_feature_importance.png")


# -----------------------------------------------------
# FIGURE 6 – Authentication Decisions
# -----------------------------------------------------

with open("risk_policy_thresholds.txt") as f:
    tau1, tau2 = map(float, f.read().split(","))

def decision(r):
    if r < tau1:
        return "ALLOW"
    elif r < tau2:
        return "STEP_UP"
    else:
        return "BLOCK"

df["Decision"] = df["Risk_Score"].apply(decision)

decision_counts = df["Decision"].value_counts()
print("\nAuthentication Decisions Totals:")
print(f"Total ALLOW: {decision_counts.get('ALLOW', 0)}")
print(f"Total STEP_UP: {decision_counts.get('STEP_UP', 0)}")
print(f"Total BLOCK: {decision_counts.get('BLOCK', 0)}")

plt.figure()
df["Decision"].value_counts().plot(kind="bar")
plt.title("Authentication Decision Distribution")
plt.xlabel("Decision")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig6_policy_decisions.png")


# -----------------------------------------------------
# FIGURE 7 – Trust Evolution Example
# -----------------------------------------------------

df["Time_of_Access"] = pd.to_datetime(df["Time_of_Access"])

example_user = df["User_ID"].iloc[0]

user_df = df[df["User_ID"]==example_user].sort_values("Time_of_Access")

plt.figure()
plt.plot(user_df["Time_of_Access"], user_df["Trust_Score"])
plt.title("Trust Evolution Example")
plt.xlabel("Time")
plt.ylabel("Trust Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fig7_trust_evolution.png")


# -----------------------------------------------------
# FIGURE 8 – Trust Heatmap
# -----------------------------------------------------

df["Day"] = df["Time_of_Access"].dt.date

heat = df.groupby(["User_ID","Day"])["Trust_Score"].mean().reset_index()

pivot = heat.pivot(index="User_ID", columns="Day", values="Trust_Score")

pivot = pivot.head(50)

plt.figure(figsize=(12,8))
sns.heatmap(pivot, cmap="viridis")
plt.title("User Trust Heatmap")
plt.xlabel("Day")
plt.ylabel("User")
plt.tight_layout()
plt.savefig("fig8_trust_heatmap.png")


# -----------------------------------------------------
# FIGURE 9 – Risk vs Trust Phase Diagram
# -----------------------------------------------------

plt.figure()

benign = df[df["Label"]=="Benign"]
attack = df[df["Label"]=="Attack"]

plt.scatter(benign["Risk_Score"], benign["Trust_Score"], alpha=0.3, label="Benign")
plt.scatter(attack["Risk_Score"], attack["Trust_Score"], alpha=0.3, label="Attack")

plt.axvline(x=tau1, linestyle="--")
plt.axvline(x=tau2, linestyle="--")

plt.xlabel("Risk Score")
plt.ylabel("Trust Score")
plt.title("Risk–Trust Phase Diagram")
plt.legend()

plt.tight_layout()
plt.savefig("fig9_risk_trust_phase.png")


# -----------------------------------------------------
# FIGURE 10 – User Friction Distribution
# -----------------------------------------------------

benign = df[df["Label"]=="Benign"]

plt.figure()
benign["Decision"].value_counts().plot(kind="bar")
plt.title("User Friction Distribution")
plt.xlabel("Decision")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("fig10_user_friction.png")

travel=df.groupby(["Geo_Location","Travel_Status"]).size().unstack()

travel.plot(kind="bar",stacked=True)

plt.title("User Travel Distribution")

plt.savefig("travel_patterns.png")


# Detection Delay

delays = []

# ---------------------------------------
# Compute detection delays per attacker IP
# ---------------------------------------

# for ip, g in df.groupby("IP_Address"):

#     g = g.sort_values("Time_of_Access")

#     attacks = g[g["Label"] == "Attack"]

#     if len(attacks) == 0:
#         continue

#     attack_start = attacks["Time_of_Access"].min()

#     detections = g[
#         (g["Risk_Score"] >= tau2) &
#         (g["Time_of_Access"] >= attack_start)
#     ]

#     if len(detections) == 0:
#         continue

#     detect_time = detections["Time_of_Access"].min()

#     delay = (detect_time - attack_start).total_seconds()

#     delays.append(delay)

# delays = [d for d in delays if d<3600]

# delays = np.array(delays)

# # ---------------------------------------
# # FIGURE 1: Detection Delay Histogram
# # ---------------------------------------

# plt.figure(figsize=(7,5))

# plt.hist(delays, bins=40, alpha=0.75)

# plt.xlabel("Detection Delay (seconds)")
# plt.ylabel("Number of Attack Campaigns")
# plt.title("Detection Delay Across Attack Campaigns")

# plt.grid()

# plt.savefig("figure_detection_delay_histogram.png", dpi=300)
# plt.close()

# # ---------------------------------------
# # FIGURE 2: Detection Delay CDF
# # ---------------------------------------

# sorted_delays = np.sort(delays)
# cdf = np.arange(len(sorted_delays)) / float(len(sorted_delays))

# plt.figure(figsize=(7,5))

# plt.plot(sorted_delays, cdf)

# plt.xlabel("Detection Delay (seconds)")
# plt.ylabel("Cumulative Probability")
# plt.title("CDF of Detection Delay")

# plt.grid()

# plt.savefig("figure_detection_delay_cdf.png", dpi=300)
# plt.close()


# ---------------------------------------
# Compute detection delays per attacker IP
# ---------------------------------------

for ip, g in df.groupby("IP_Address"):

    g = g.sort_values("Time_of_Access").reset_index(drop=True)

    g["event_order"] = range(len(g))

    attacks = g[g["Label"] == "Attack"]

    if len(attacks) == 0:
        continue

    attack_start = attacks["event_order"].min()

    detections = g[
        (g["Risk_Score"] >= tau2) &
        (g["event_order"] >= attack_start)
    ]

    if len(detections) == 0:
        continue

    detect_event = detections["event_order"].min()

    # delay = (detect_time - attack_start).total_seconds()
    
    # attack_index = attacks.index.min()
    # detect_index = detections.index.min()
    
    delay = detect_event - attack_start

    delays.append(delay)

# delays = [d for d in delays if d<3600]

delays = np.array(delays)


if len(delays)>0:
    # print("Avg delay:", sum(delays)/len(delays))
    print("Average delay:", np.mean(delays))
    print("Median delay:", np.median(delays))
    print("Max delay:", np.max(delays))
    print("95th percentile delay:", np.percentile(delays, 95))
    print("Immediate detection rate:", sum(d == 0 for d in delays)/len(delays))

# ---------------------------------------
# FIGURE 1: Detection Delay Histogram
# ---------------------------------------

plt.figure(figsize=(7,5))

bins = np.arange(-0.5,6.5,1)

plt.hist(delays, bins=bins, edgecolor='black')
plt.xticks(range(0,6))

plt.xlabel("Events Before Detection")
plt.ylabel("Number of Attack Campaigns")
plt.title("Detection Delay ( Events Before Detection") 

plt.grid()

plt.savefig("figure_detection_delay_events.png", dpi=300)
plt.close()

# ---------------------------------------
# FIGURE 2: Detection Delay CDF
# ---------------------------------------

sorted_delays = np.sort(delays)
cdf = np.arange(len(sorted_delays)) / float(len(sorted_delays))

plt.figure(figsize=(7,5))

plt.step(sorted_delays, cdf, where = "post")
plt.xlim(-0.5,5)

plt.xlabel("Events Before Detection")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Detection Delay")

plt.grid()

plt.savefig("figure_detection_delay_cdf_events.png", dpi=300)
plt.close()

print("All figures generated successfully.")


#Attribute weights

