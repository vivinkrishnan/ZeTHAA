import pandas as pd
import numpy as np

df=pd.read_csv("synthetic_authentication_dataset.csv")

df["Time_of_Access"]=pd.to_datetime(df["Time_of_Access"])

with open("risk_policy_thresholds.txt") as f:
    tau1, tau2 = map(float, f.read().split(","))

# detected = df[df["Risk_Score"] >= tau2]

delays=[]

for ip,g in df.groupby("IP_Address"):
    
    g = g.sort_values("Time_of_Access")

    attacks=g[g["Label"]=="Attack"]

    if len(attacks)==0:
        continue

    start=attacks["Time_of_Access"].min()

    detections = g[
        (g["Risk_Score"] >= tau2) &
        (g["Time_of_Access"] >= start)
    ]
    
    if len(detections)==0:
        continue
    
    detect=detections["Time_of_Access"].min()

    delay = (detect-start).total_seconds()

    delays.append((detect-start).total_seconds())
    
delays = [d for d in delays if d<3600]
    
if len(delays)>0:
    print("Avg delay:", sum(delays)/len(delays))
    print("Average delay:", np.mean(delays))
    print("Median delay:", np.median(delays))
    print("Max delay:", np.max(delays))
    print("95th percentile delay:", np.percentile(delays, 95))
    print("Immediate detection rate:", sum(d == 0 for d in delays)/len(delays))
              
else:
    print("No detections recorded")
