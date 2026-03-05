import numpy as np, pandas as pd, joblib
from pathlib import Path

models = Path("models")
ood    = joblib.load(models / "ood_detector.pkl")
thresh = joblib.load(models / "ood_threshold.pkl")
ohe    = joblib.load(models / "ohe_encoder.pkl")
fnames = joblib.load(models / "feature_names.pkl")

def build_row(q):
    r = dict(q)
    r["Miles_Per_Exp"]   = r["Annual_Miles"] / (r["Driving_Exp"] + 1)
    r["Total_Incidents"] = r["Prev_Accidents"] + r["Prev_Citations"]
    r["Age_Exp_Gap"]     = r["Driver_Age"] - r["Driving_Exp"] - 16
    vdf  = pd.DataFrame([[r["Veh_Usage"]]], columns=["Veh_Usage"])
    venc = ohe.transform(vdf)
    for col, val in zip(ohe.get_feature_names_out(["Veh_Usage"]), venc[0]):
        r[col] = val
    return pd.DataFrame([r])[fnames].astype(float)

demos = [
    ("High-Risk (accident+citation, age22, exp3, business, 40K)",
     {"Prev_Accidents":1,"Prev_Citations":1,"Driving_Exp":3,"Driver_Age":22,"Annual_Miles":40000,"Veh_Usage":"Business"}),
    ("Low-Risk  (clean, age42, exp20, pleasure, 10K)",
     {"Prev_Accidents":0,"Prev_Citations":0,"Driving_Exp":20,"Driver_Age":42,"Annual_Miles":10000,"Veh_Usage":"Pleasure"}),
    ("Med-Risk  (1 citation, age30, exp8, commute, 28K)",
     {"Prev_Accidents":0,"Prev_Citations":1,"Driving_Exp":8,"Driver_Age":30,"Annual_Miles":28000,"Veh_Usage":"Commute"}),
    ("CORRUPT   (age=-5, miles=9999999)",
     {"Prev_Accidents":0,"Prev_Citations":0,"Driving_Exp":0,"Driver_Age":-5,"Annual_Miles":9999999,"Veh_Usage":"Pleasure"}),
]

print(f"Threshold (0.10th pct): {thresh:.6f}\n")
for label, q in demos:
    sc = float(ood.score_samples(build_row(q))[0])
    flag = "FLAGGED" if sc < thresh else "OK"
    print(f"  [{flag:7s}]  score={sc:+.6f}   {label}")
