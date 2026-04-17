import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("synthetic_authentication_dataset.csv")

df["Attack_Flag"] = (df["Label"]=="Attack").astype(int)

signals = [
"Geo_Anomaly",
"Travel_Anomaly",
"Timezone_Anomaly",
"Temporal_Anomaly",
"Fingerprint_Mismatch",
"Repeated_Attacker_IP",
"Token_Context_Mismatch"
]

baseline = df["Attack_Flag"].mean()

df["Geo_Anomaly"] = (df["Geo_Location"] != df.groupby("User_ID")["Geo_Location"].transform("first"))

df["Travel_Anomaly"] = df["Travel_Status"].astype(bool)

df["Timezone_Anomaly"] = (df["Timezone_Shift"] != 0)

df["Temporal_Anomaly"] = (df["Login_Hour"] < 6) | (df["Login_Hour"] > 22)

df["Fingerprint_Mismatch"] = df["User_Agent"].str.contains("Python|curl|bot", case=False, na=False)

df["Token_Context_Mismatch"] = (
    df.groupby("Session_Token_ID")["Geo_Location"].transform("nunique") > 1
)

df["Repeated_Attacker_IP"] = (
    df.groupby("IP_Address")["Attack_Flag"].transform("sum") > 3
)

rows = []

for signal in signals:
    
    prob = df.groupby(signal)["Attack_Flag"].mean()
    p_true = prob.get(True,0)
    
    lift = p_true / baseline
    
    rows.append({"Signal": signal, "P(Attack|Signal)": p_true, "Baseline_Attack_Rate": baseline, "Lift": lift})
    
results = pd.DataFrame(rows)

results.sort_values("Lift", ascending=False, inplace=True)

print(results)

plt.figure(figsize=(8,5))

plt.barh(results["Signal"], results["Lift"], color="salmon")
plt.xlabel("Lift")
plt.title("Signal Lift Analysis")

plt.gca().invert_yaxis()

plt.grid()

plt.savefig("signal_lift.png", dpi=300)
