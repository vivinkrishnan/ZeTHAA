import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score

df=pd.read_csv("synthetic_authentication_dataset.csv")

df["Attack_Flag"]=(df["Label"]=="Attack").astype(int)

attributes=[
    "Device_Model",
    "Geo_Location",
    "Travel_Status",
    "Timezone_Shift",
    "Platform",
    "OS_Version",
    "IP_Address",
    "Application_Version",
    "Application_Authenticity_Data",
    "User_Agent",
    "Screen_Resolution",
    "Request_Type"
]

results = []
for attr in attributes:
    ig = mutual_info_score(df[attr],df["Attack_Flag"])
    
    attack_probs = df.groupby(attr)["Attack_Flag"].mean()
    
    penalty = attack_probs.max()
    
    results.append({"Attribute":attr,"Information_Gain":ig, "Penalty":penalty})
    
weights = pd.DataFrame(results)

weights["Weight"] = weights["Information_Gain"] / weights["Information_Gain"].sum()

weights.sort_values("Weight", ascending=False, inplace=True)

weights.to_csv("attribute_weights_penalties.csv", index=False)

print(weights)