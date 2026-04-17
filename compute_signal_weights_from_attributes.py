import pandas as pd

attr = pd.read_csv("attribute_entropy_bayesian_analysis.csv")

mapping = {
    "Geo_Anomaly": ["Geo_Location"],
    "Travel_Anomaly": ["Travel_Status"],
    "Timezone_Anomaly": ["Timezone_Shift"],
    "Temporal_Anomaly": ["Login_Hour"],
    "Fingerprint_Mismatch": ["User_Agent","Device_Model","Screen_Resolution"],
    "Repeated_Attacker_IP": ["IP_Address"],
    "Application_Tampering": ["Application_Version","Application_Authenticity_Data"],
    "Session_Token_Mismatch": ["Session_Token_ID"]           
}

rows = []

for signal, attrs in mapping.items():
    subset = attr[attr["Attribute"].isin(attrs)]
    entropy_weight = subset["Entropy_Weight"].sum()
    bayesian_weight = subset["Bayesian_Weight"].mean()
    penalty = subset["Penalty"].max()
    
    rows.append({"Signal": signal, "Attributes": ", ".join(attrs), "Entropy_Weight": entropy_weight, "Bayesian_Weight": bayesian_weight, "Penalty": penalty})
    
signals = pd.DataFrame(rows)

signals["Normalized_Weight"] = signals["Entropy_Weight"] / signals["Entropy_Weight"].sum()

signals.sort_values("Normalized_Weight", ascending=False, inplace=True)

signals.to_csv("signal_weights_from_attributes.csv", index=False)

print(signals)