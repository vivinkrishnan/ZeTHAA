import pandas as pd
import random
import uuid
import sys
from datetime import datetime, timedelta

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

NUM_USERS = 500
NUM_SESSIONS = 50000
DATASET_DAYS = 30
ATTACK_RATIO = float(sys.argv[1]) if len(sys.argv)>1 else 0.3
NUM_ATTACK_CAMPAIGNS = 150
user_ip_history = {}


START_DATE = datetime(2026,1,1)

TARGET_SIGNAL_RATES = {
"Geo_Anomaly":0.08,
"Travel_Anomaly":0.05,
"Timezone_Anomaly":0.03,
"Temporal_Anomaly":0.07,
"Fingerprint_Mismatch":0.06,
"Token_Context_Mismatch":0.04,
"Repeated_Attacker_IP":0.05
}

ATTACK_SIGNAL_MULTIPLIER = {
"Geo_Anomaly":2.0,
"Travel_Anomaly":1.5,
"Timezone_Anomaly":1.5,
"Temporal_Anomaly":1.5,
"Fingerprint_Mismatch":2.5,
"Token_Context_Mismatch":3.0,
"Repeated_Attacker_IP":2.5
}

signal_counts = {k:0 for k in TARGET_SIGNAL_RATES}
total_events = 0

# ----------------------------------------------------
# LOCATION MODEL
# ----------------------------------------------------

CITIES = {
"Bangalore":{"country":"IN","tz":5.5},
"Chennai":{"country":"IN","tz":5.5},
"Delhi":{"country":"IN","tz":5.5},
"Mumbai":{"country":"IN","tz":5.5},
"London":{"country":"UK","tz":0},
"Berlin":{"country":"DE","tz":1},
"NewYork":{"country":"US","tz":-5},
"Tokyo":{"country":"JP","tz":9}
}

# ----------------------------------------------------
# DEVICES
# ----------------------------------------------------

DEVICE_CATALOG = {
"GalaxyS23":("Samsung","Android"),
"Pixel7":("Google","Android"),
"RedmiNote":("Xiaomi","Android"),
"iPhone13":("Apple","iOS")
}

APP_VERSIONS=["5.1","5.2","5.3","5.4"]

VERSION_ROLLOUT={
"5.1":(0,7),
"5.2":(5,15),
"5.3":(14,23),
"5.4":(20,30)
}

# ----------------------------------------------------
# ATTACK TYPES
# ----------------------------------------------------

ATTACK_TYPES=[
"Credential_Theft",
"Bot_Login",
"Session_Hijack",
"Emulator",
"Impossible_Travel",
"Application_Tampering",
"Access_Without_Login",
"Device_Spoofing",
"Device_Cloning",
"Token_Theft"
]

ATTACK_SIGNAL_PATTERNS={
"Credential_Theft":["Geo_Anomaly","Fingerprint_Mismatch"],
"Bot_Login":["Temporal_Anomaly","Repeated_Attacker_IP"],
"Session_Hijack":["Token_Context_Mismatch","Geo_Anomaly"],
"Device_Spoofing":["Fingerprint_Mismatch","Repeated_Attacker_IP"],
"Impossible_Travel":["Geo_Anomaly","Timezone_Anomaly"],
"Token_Theft":["Token_Context_Mismatch","Geo_Anomaly"]
}

BENIGN_FLOWS=[
["login","resource"],
["login","challenge","resource"],
["login","resource","resource"]
]

ATTACK_FLOWS=[
["login"],
["login","login"],
["login","challenge"],
["login","challenge","resource"],
["resource"]
]

APP_AUTH={v:"AHD-"+str(random.randint(100000,999999)) for v in APP_VERSIONS}

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

def choose_version(day):
    valid=[v for v,(s,e) in VERSION_ROLLOUT.items() if s<=day<=e]
    return random.choice(valid if valid else APP_VERSIONS)

def random_ip():
    return ".".join(str(random.randint(1,255)) for _ in range(4))

def fingerprint():
    return {
        "User_Agent":random.choice(["Mozilla Android","Mobile Safari","Mozilla iPhone"]),
        "Screen_Resolution":random.choice(["1080x2400","1170x2532"])
    }

def token():
    return "TKN-"+str(uuid.uuid4())[:12]

# ----------------------------------------------------
# USER PROFILES
# ----------------------------------------------------

users={}

for i in range(NUM_USERS):

    uid=f"U{i:04}"

    home=random.choice(list(CITIES.keys())[:4])

    preferred_device=random.choice(list(DEVICE_CATALOG.keys()))

    users[uid]={
        "home_city":home,
        "current_city":home,
        "traveling":False,
        "travel_end_day":-1,
        "devices":[preferred_device],
        "weekday_login_mean":random.randint(7,10),
        "weekend_login_mean":random.randint(10,12),
        "os_version":14,
        "trust":0.5
    }

# ----------------------------------------------------
# ATTACK CAMPAIGNS
# ----------------------------------------------------

campaigns=[{
"ip":random_ip(),
"device":random.choice(list(DEVICE_CATALOG.keys())),
"type":random.choice(ATTACK_TYPES)
} for _ in range(NUM_ATTACK_CAMPAIGNS)]

records=[]

# ----------------------------------------------------
# SESSION GENERATION
# ----------------------------------------------------

for s in range(NUM_SESSIONS):

    user=random.choice(list(users.keys()))
    profile=users[user]

    attack=random.random()<ATTACK_RATIO
    campaign=random.choice(campaigns) if attack and random.random()<0.5 else None
    attack_type=campaign["type"] if campaign else random.choice(ATTACK_TYPES)

    day=random.randint(0,DATASET_DAYS-1)

    date=START_DATE+timedelta(days=day)

    weekday=date.weekday()

    is_weekend=weekday>=5

# ----------------------------------------------------
# TRAVEL MODEL
# ----------------------------------------------------

    travel_ratio=signal_counts["Travel_Anomaly"]/max(total_events,1)

    if not profile["traveling"]:

        if travel_ratio < TARGET_SIGNAL_RATES["Travel_Anomaly"]:

            if random.random()<0.005:

                travel_city=random.choice(
                    [c for c in CITIES.keys() if c!=profile["home_city"]]
                )

                profile["current_city"]=travel_city
                profile["traveling"]=True
                profile["travel_end_day"]=day+random.randint(2,4)

    if profile["traveling"] and day>profile["travel_end_day"]:

        profile["traveling"]=False
        profile["current_city"]=profile["home_city"]

    city=profile["current_city"]
    travel_flag=profile["traveling"]

    tz_shift=CITIES[city]["tz"]-CITIES[profile["home_city"]]["tz"]

# ----------------------------------------------------
# LOGIN TIME MODEL
# ----------------------------------------------------

    login_mean=profile["weekday_login_mean"] if weekday<5 else profile["weekend_login_mean"]

    login_hour=int(random.gauss(login_mean+tz_shift,1.2))

# ----------------------------------------------------
# DEVICE LIFECYCLE
# ----------------------------------------------------

    if random.random()<0.01:
        profile["devices"].append(random.choice(list(DEVICE_CATALOG.keys())))

    device=random.choice(profile["devices"])
    manufacturer,platform=DEVICE_CATALOG[device]

# ----------------------------------------------------
# OS UPGRADE
# ----------------------------------------------------

    if random.random()<0.02:
        profile["os_version"]+=1

    os_version=profile["os_version"]

# ----------------------------------------------------
# APPLICATION VERSION
# ----------------------------------------------------

    version=choose_version(day)
    auth=APP_AUTH[version]

    integrity=False
    if attack_type=="Application_Tampering":
        auth="AHD-"+str(random.randint(100000,999999))
        integrity=True

# ----------------------------------------------------
# SESSION FLOW
# ----------------------------------------------------

    flow=random.choice(ATTACK_FLOWS if attack else BENIGN_FLOWS)

    session_id=f"S{s:06}"

    base_fp=fingerprint()
    session_token=token()

    timestamp=START_DATE+timedelta(days=day,hours=login_hour)

# ----------------------------------------------------
# ATTACK SIGNAL FLAGS
# ----------------------------------------------------

    signal_flags={k:False for k in TARGET_SIGNAL_RATES}

    if attack and attack_type in ATTACK_SIGNAL_PATTERNS:
        for sig in ATTACK_SIGNAL_PATTERNS[attack_type]:
            signal_flags[sig]=True
            
    # if not attack and random.random() < TARGET_SIGNAL_RATES["Repeated_Attacker_IP"]:
    #     signal_flags["Repeated_Attacker_IP"] = True
    
    if not attack and signal_flags["Repeated_Attacker_IP"]:
        print("Benign IP reused")
    
    for sig, rate in TARGET_SIGNAL_RATES.items():
        if signal_flags[sig]:
            continue
        
        if attack:
            activate_prob = min(rate *ATTACK_SIGNAL_MULTIPLIER[sig],0.9)
        else:
            activate_prob = rate
        
        if random.random() < activate_prob:
            signal_flags[sig] = True
    

# ----------------------------------------------------
# EVENT GENERATION
# ---------------------------------------------------- 

    # print("Signal_flags", signal_flags)

    for event_index,request_type in enumerate(flow):

        event_time=timestamp+timedelta(seconds=event_index*5)

        event_city=city
        tok=session_token
        fp=base_fp.copy()

        if signal_flags["Fingerprint_Mismatch"] and event_index>0:
            fp["User_Agent"]="Python Requests"

        if signal_flags["Token_Context_Mismatch"] and event_index>0 and random.random()<0.5:
            event_city=random.choice(list(CITIES.keys()))

        if signal_flags["Geo_Anomaly"] and random.random() < 0.7:
            event_city=random.choice(
                [c for c in CITIES.keys() if c!=profile["home_city"]]
            )

        # ip_addr=campaign["ip"] if campaign else random_ip()
        
        if signal_flags["Repeated_Attacker_IP"]:
            if campaign:
                ip_addr=campaign["ip"]
            else:
                ip_addr=random.choice(user_ip_history.get(user, [random_ip()]))
        else:
            ip_addr=random_ip()
        
        user_ip_history.setdefault(user, []).append(ip_addr)    

        # risk=random.uniform(0.1,0.4) if attack else random.uniform(0,0.2)
        
        # if attack:
        #     risk = random.uniform(0.05,0.4)
        # else:
        #     risk = random.uniform(0,0.25)
        
        base_risk = random.uniform(0.01, 0.10)
        
        risk = base_risk
        
        # if signal_flags["Geo_Anomaly"]:
        #     base_risk += 0.1
        # if signal_flags["Fingerprint_Mismatch"]:
        #     base_risk += 0.15
        # if signal_flags["Token_Context_Mismatch"]:
        #     base_risk += 0.2
        # if signal_flags["Repeated_Attacker_IP"]:
        #     base_risk += 0.15
        
        if signal_flags["Geo_Anomaly"]:
            risk += 0.08
        if signal_flags["Travel_Anomaly"]:
            risk += 0.05
        if signal_flags["Timezone_Anomaly"]:
            risk += 0.04
        if signal_flags["Temporal_Anomaly"]:
            risk += 0.06
        if signal_flags["Fingerprint_Mismatch"]:
            risk += 0.14 if attack else 0.10            
        if signal_flags["Token_Context_Mismatch"]:
            risk += 0.16 if attack else 0.12
        if signal_flags["Repeated_Attacker_IP"]:
            risk += 0.12 if attack else 0.08
            
        if attack:
            if random.random() < 0.7:
                risk += 0.025
                
        # risk += random.uniform(-0.01,0.02)
        risk += random.uniform(-0.005,0.01)
            
        risk = max(0,min(risk, 0.4))

        trust=profile["trust"]
        trust=trust*0.95-risk+0.1*(1-risk)
        trust=max(0,min(1,trust))
        profile["trust"]=trust
        
        attack_probability = 0.05
        
        if attack:
            label = "Attack"
            attack_type = attack_type            
        else:
            label = "Benign"
            attack_type = "None"
        
        if not attack and random.random() < 0.01:
            signal_flags ["Fingerprint_Mismatch"] = True
            
        if not attack and random.random() < 0.01:
            signal_flags ["Token_Context_Mismatch"] = True
            
        if not attack and random.random() < 0.02:
            signal_flags["Repeated_Attacker_IP"] = True
         
        # if signal_flags["Geo_Anomaly"]:
        #     attack_probability += 0.40

        # if signal_flags["Fingerprint_Mismatch"]:
        #     attack_probability += 0.35

        # if signal_flags["Token_Context_Mismatch"]:
        #     attack_probability += 0.35

        # if signal_flags["Repeated_Attacker_IP"]:
        #     attack_probability += 0.30

        # label = "Attack" if random.random() < attack_probability else "Benign"
        
        # label = "Attack" if attack else "Benign"

        records.append({

        "Session_ID":session_id,
        "Event_ID":event_index,
        "User_ID":user,

        "Time_of_Access":event_time,
        "Weekday":weekday,
        "Is_Weekend":is_weekend,
        "Login_Hour":login_hour,

        "Device_Model":device,
        "Manufacturer":manufacturer,
        "Platform":platform,
        "OS_Version":os_version,

        "Geo_Location":event_city,
        "Travel_Status":travel_flag,
        "Timezone_Shift":tz_shift,

        "IP_Address":ip_addr,

        "Application_Version":version,
        "Application_Authenticity_Data":auth,
        **fp,

        "Session_Token_ID":tok,

        "Request_Type":request_type,

        "Risk_Score":risk,
        "Trust_Score":trust,

        # "Attack_Type":attack_type if attack else "None",
        "Attack_Type":attack_type,
        "Label": label
        })

        if travel_flag:
            signal_counts["Travel_Anomaly"]+=1

        total_events+=1

# ----------------------------------------------------
# SAVE DATASET
# ----------------------------------------------------

df=pd.DataFrame(records)
print(df[df["Attack_Type"]!="None"]["Label"].value_counts())
print(df[df["Label"]=="Benign"]["Attack_Type"].value_counts())

print(df["Label"].value_counts(normalize=True))
# print(df[df["Repeated_Attacker_IP"]]["Label"].value_counts())
# print(df.groupby("Fingerprint_Mismatch")["Attack_Flag"].mean())
# print(df.groupby("Token_Context_Mismatch")["Attack_Flag"].mean())
print(df[(df["Label"]=="Benign") & (df["Attack_Type"]!="None") ])
print(df[df["Label"]=="Benign"]["Risk_Score"].quantile([0.75,0.9]))

df.to_csv("synthetic_authentication_dataset.csv",index=False)

print("Dataset generated:",len(df))