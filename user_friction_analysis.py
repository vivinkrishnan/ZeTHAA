import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler




df=pd.read_csv("synthetic_authentication_dataset.csv")

with open("risk_policy_thresholds.txt") as f:
    tau1_actual,tau2_actual=map(float,f.read().split(","))

# df["Decision"]=df["Risk_Score"].apply(lambda r:"ALLOW" if r<tau1 else("STEP_UP" if r<tau2 else"BLOCK"))

# benign=df[df["Label"]=="Benign"]

# print("Step-Up Rate",(benign["Decision"]=="STEP_UP").mean())
# print("False Block Rate",(benign["Decision"]=="BLOCK").mean())

df["Attack_Flag"] = (df["Label"]=="Attack").astype(int) 

df["Geo_Anomaly"]=df.groupby("User_ID")["Geo_Location"].transform(lambda x:x!=x.mode()[0])
df["Fingerprint_Mismatch"]=df.groupby("Session_ID")["User_Agent"].transform("nunique")>1
df["Token_Context_Mismatch"]=df.groupby("Session_Token_ID")["Geo_Location"].transform("nunique")>1
df["Repeated_Attacker_IP"]=df.groupby("IP_Address")["Attack_Flag"].transform("sum")>3
df["Timezone_Anomaly"] = abs(df["Timezone_Shift"]) > 3
df["Travel_Anomaly"] = df["Travel_Status"]
df["Temporal_Anomaly"] = (
    (df["Is_Weekend"] == True) &
    (df["Login_Hour"] < 6)
)

features = [
    "Geo_Anomaly",
    "Travel_Anomaly",
    "Timezone_Anomaly",
    "Temporal_Anomaly",
    "Fingerprint_Mismatch",
    "Token_Context_Mismatch",
    "Repeated_Attacker_IP"    
]

cost_map={
    "ALLOW": 1,
    "STEP_UP": 5,
    "BLOCK": 10
}
def calibrate(prob):
    scale = tau2_actual/np.percentile(prob, 90)
    return np.clip(prob*scale, 0, 1)


def decision(r):
    if r<tau1_actual:
        return "ALLOW"
    elif r<tau2_actual:
        return "STEP_UP"
    else:
        return "BLOCK"
    
def evaluate_baseline(prob_cal):
    df["Decision"]=df[prob_cal].apply(lambda r:"ALLOW" if r<tau1_actual else("STEP_UP" if r<tau2_actual else"BLOCK"))
    df["Cost"] = df["Decision"].map(cost_map)
    avg_cost = df["Cost"].mean()
    
    attack_df = df[df["Attack_Flag"]==1]
    detected = attack_df[attack_df["Decision"]=="BLOCK"]
    recall = len(detected) / len(attack_df)
    return avg_cost, recall
    

df["Decision_actual"]=df["Risk_Score"].apply(decision)

df["Auth_Cost_actual"] = df["Decision_actual"].map(cost_map)
current_cost = df["Auth_Cost_actual"].mean()

table = pd.crosstab(df["Decision_actual"], df["Label"])
print("\nPolicy Decision Matrix:\n", table)

total = len(df)

step_up_rate = (df["Decision_actual"]=="STEP_UP").mean()
false_block_rate = len(df[(df["Decision_actual"]=="BLOCK") & (df["Label"]=="Benign")]) / total
attack_block_rate = len(df[(df["Decision_actual"]=="BLOCK") & (df["Label"]=="Attack")]) / len(df[df["Label"]=="Attack"])

print("\nUser Friction Metrics:")
print(f"Step-Up Rate: {step_up_rate:.4f}")
print(f"False Block Rate: {false_block_rate:.4f}")
print(f"Attack Block Rate: {attack_block_rate:.4f}")
print(f"Average Authentication Cost: ${current_cost:.2f}")
# print(f'Attack Detection Recall: {recall:.4f}')



x = df[features].astype(int)
y = df["Attack_Flag"]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

lr = LogisticRegression()
lr.fit(x_scaled,y)
df["LR_Prob"] = lr.predict_proba(x_scaled)[:,1]

rf = RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x,y)
df["RF_Prob"] = rf.predict_proba(x)[:,1]

lr_cost,lr_recall = evaluate_baseline("LR_Prob")
rf_cost,rf_recall = evaluate_baseline("RF_Prob")


results = []

taus = np.linspace(0.05,0.30,20)

for tau in taus:
    for tau2 in taus:
        if tau2 <= tau:
            continue
        
        def decision_in(r):
            if r<tau:
                return "ALLOW"
            elif r<tau2:
                return "STEP_UP"
            else:
                return "BLOCK"
        
        
        df["Decision"]=df["Risk_Score"].apply(decision_in)
        df["Cost"] = df["Decision"].map(cost_map)
        avg_cost = df["Cost"].mean()
        
        attack_df = df[df["Attack_Flag"]==1]
        detected = attack_df[attack_df["Decision"]=="BLOCK"]
        recall = len(detected) / len(attack_df)
        results.append((avg_cost,recall))
        

res_df = pd.DataFrame(results, columns=["Cost", "Recall"])
plt.figure(figsize=(7,5))
plt.scatter(res_df["Cost"], res_df["Recall"], alpha=0.6)
plt.scatter(current_cost, attack_block_rate, marker="x", s=100, label="Proposed Framework")
# plt.scatter(lr_cost, lr_recall, marker="o", s=100, label="Logistic Regression")
# plt.scatter(rf_cost, rf_recall, marker="s", s=100, label="Random Forest")
plt.legend()
plt.xlabel("Average Authentication Cost")
plt.ylabel("Attack Detection Recall")
plt.title("Cost vs. Attack Detection Recall for Different Thresholds")
plt.grid(True)
plt.savefig("cost_vs_recall.png", dpi=300)
plt.show()


# palette = {"ALLOW": "green", "STEP_UP": "orange", "BLOCK": "red"}
# plt.figure(figsize=(7,5))
# sns.histplot(data=df, x="Risk_Score", hue="Decision", bins=50, palette=palette)
# plt.title("Risk Score Distribution by Policy Decision")
# plt.xlabel("Risk Score")
# plt.ylabel("Frequency")
# plt.savefig("risk_score_distribution_by_decision.png", dpi=300)




plt.show()