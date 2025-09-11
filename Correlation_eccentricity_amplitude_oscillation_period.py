import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob, os
from scipy.stats import linregress
from scipy.signal import find_peaks
from tqdm import tqdm

loc = "C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Results"
out_dir = "C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Power_laws/f(eccentricity)=amplitude"
os.makedirs(out_dir, exist_ok=True)

dis_files = glob.glob(os.path.join(loc, "eccentricities_(*).csv"))

indents = [f.replace(loc + "\\eccentricities_(", "").replace(").csv", "") for f in dis_files]

indi = indents[29:]

results = {
    "neg1_before": [],
    "neg1_after": [],
    "pos1_before": [],
    "pos1_after": []
}

for i in tqdm(indents):
    ecc_file = f"{loc}/eccentricities_({i}).csv"
    wobble_file = f"C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Treated_eccs/eccentricity_wobble_({i}).csv"
    xy_file = f"{loc}/xy_({i}).csv"

    df_xy = pd.read_csv(xy_file, low_memory=False)

    df_xy["x"] = pd.to_numeric(df_xy["x"], errors="coerce")
    df_xy["y"] = pd.to_numeric(df_xy["y"], errors="coerce")

    df_xy = df_xy.dropna(subset=["x", "y"])

    r = np.sqrt(df_xy["x"]**2 + df_xy["y"]**2)


    df_ecc = pd.read_csv(ecc_file)
    t_full = df_ecc["years"].values

    aphelia_idx, _ = find_peaks(r)
    t_aphelia = t_full[aphelia_idx]
    r_aphelia = r[aphelia_idx]

    R_out = 0.02072
    engulf_idx = np.argmax(r_aphelia < R_out)
    t_engulf = t_aphelia[engulf_idx]
    
    #to compute the oscillation periods use the same code but replace amplitude with wobble period [min]
    df_wob = pd.read_csv(wobble_file)
    df_wob["eccentricity"] = df_wob["time [years]"].apply(
        lambda t: df_ecc.iloc[(df_ecc["years"] - t).abs().argmin()]["eccentricity"]
    )

    df_before = df_wob[df_wob["time [years]"] < t_engulf]
    df_after  = df_wob[df_wob["time [years]"] >= t_engulf]

    def analyze_and_store(df_seg, key, ident):
        mask = (df_seg["eccentricity"] > 0) & (df_seg["amplitude"] > 0)
        if not mask.any():
            return
        x = df_seg.loc[mask, "eccentricity"].values
        y = df_seg.loc[mask, "amplitude"].values
        logx, logy = np.log(x), np.log(y)

        slope, intercept, r_value, _, _ = linregress(logx, logy)
        results[key].append({
            "identifier": ident,
            "slope": slope,
            "intercept": intercept,
            "R2": r_value**2
        })

    direction = int(i.split("_")[1])  
    if direction == -1:
        analyze_and_store(df_before, "neg1_before", i)
        analyze_and_store(df_after,  "neg1_after", i)
    else:
        analyze_and_store(df_before, "pos1_before", i)
        analyze_and_store(df_after,  "pos1_after", i)


for key, recs in results.items():
    if recs: 
        out_path = os.path.join(out_dir, f"fit_amplitude_results_{key}.csv")
        pd.DataFrame(recs).to_csv(out_path, index=False)
        print(f"Saved {out_path}")
