import pandas as pd

# -------------------------------------------------
# Load dataset
# -------------------------------------------------

df = pd.read_csv("synthetic_authentication_dataset.csv")

df["Time_of_Access"] = pd.to_datetime(df["Time_of_Access"])

df["Attack_Flag"] = (df["Label"] == "Attack").astype(int)

# -------------------------------------------------
# Feature engineering (same as evaluation pipeline)
# -------------------------------------------------

df["Login_Hour"] = df["Time_of_Access"].dt.hour

df["Geo_Anomaly"] = df.groupby("User_ID")["Geo_Location"].transform(
    lambda x: x != x.mode()[0]
)

df["Travel_Anomaly"] = df["Travel_Status"]

df["Timezone_Anomaly"] = abs(df["Timezone_Shift"]) > 3

df["Temporal_Anomaly"] = (df["Login_Hour"] < 6) | (df["Login_Hour"] > 22)

df["Fingerprint_Mismatch"] = df.groupby("Session_ID")["User_Agent"].transform(
    "nunique"
) > 1

df["Token_Context_Mismatch"] = df.groupby("Session_Token_ID")[
    "Geo_Location"
].transform("nunique") > 1

df["Repeated_Attacker_IP"] = df.groupby("IP_Address")[
    "Attack_Flag"
].transform("sum") > 3

signals = [
"Geo_Anomaly",
"Travel_Anomaly",
"Timezone_Anomaly",
"Temporal_Anomaly",
"Fingerprint_Mismatch",
"Token_Context_Mismatch",
"Repeated_Attacker_IP"
]

print("\nContextual Signal Distribution\n")

for s in signals:

    counts = df[s].value_counts()

    total = len(df)

    ones = counts.get(True, 0) + counts.get(1, 0)

    print(f"{s}:")
    print(f"  True count : {ones}")
    print(f"  Ratio      : {ones/total:.4f}")
    print()

# -------------------------------------------------
# Cross-check with attacks
# -------------------------------------------------

print("\nSignal vs Attack Correlation\n")

for s in signals:

    attack_rate = df.groupby(s)["Attack_Flag"].mean()

    print(s)
    print(attack_rate)
    print()