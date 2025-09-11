import pandas as pd
import numpy as np
import glob, os
from scipy.stats import linregress
from scipy.signal import find_peaks
from tqdm import tqdm



loc = "C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Results"
out_dir = "C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Power_laws/f(a)=fall_rate"
os.makedirs(out_dir, exist_ok=True)

ecc_files = glob.glob(os.path.join(loc, "eccentricities_(*).csv"))
indents = [os.path.basename(f).replace("eccentricities_(", "").replace(").csv", "") for f in ecc_files]

R_out = 0.02072 


results = {
    "neg1_before": [],
    "neg1_after": [],
    "pos1_before": [],
    "pos1_after": []
}

for ident in tqdm(indents):
    ecc_file  = os.path.join(loc, f"eccentricities_({ident}).csv")
    peri_file = os.path.join(loc, f"planet_distances_({ident}).csv")
    xy_file   = os.path.join(loc, f"xy_({ident}).csv")

    if not (os.path.exists(ecc_file) and os.path.exists(peri_file) and os.path.exists(xy_file)):
        continue

    ecc_df  = pd.read_csv(ecc_file)
    peri_df = pd.read_csv(peri_file)
    df_xy   = pd.read_csv(xy_file, low_memory= False)


    df = pd.merge_asof(peri_df.sort_values("years"),
                       ecc_df.sort_values("years"),
                       on="years", direction="nearest")


    df["a [AU]"] = df["distance [AU]"] / (1 - df["eccentricity"])


    drdt = np.gradient(df["distance [AU]"], df["years"])
    df["fall_rate"] = -drdt

    df_xy["x"] = pd.to_numeric(df_xy["x"], errors="coerce")
    df_xy["y"] = pd.to_numeric(df_xy["y"], errors="coerce")

    df_xy = df_xy.dropna(subset=["x", "y"])

    r = np.sqrt(df_xy["x"]**2 + df_xy["y"]**2)
    t_full = ecc_df["years"].values
    aphelia_idx, _ = find_peaks(r)
    if not np.any(r[aphelia_idx] < R_out):
        continue
    t_engulf = t_full[aphelia_idx][np.argmax(r[aphelia_idx] < R_out)]

    df_before = df[df["years"] < t_engulf]
    df_after  = df[df["years"] >= t_engulf]

    def analyze_and_store(df_seg, key, ident):
        mask = (df_seg["a [AU]"] > 0) & (df_seg["fall_rate"] > 0)
        if not mask.any():
            return
        x = df_seg.loc[mask, "a [AU]"].values
        y = df_seg.loc[mask, "fall_rate"].values
        logx, logy = np.log(x), np.log(y)

        slope, intercept, r_value, _, _ = linregress(logx, logy)
        results[key].append({
            "identifier": ident,
            "slope": slope,
            "intercept": intercept,
            "R2": r_value**2
        })

    direction = int(ident.split("_")[1])

    if direction == -1:
        analyze_and_store(df_before, "neg1_before", ident)
        analyze_and_store(df_after,  "neg1_after", ident)
    else:
        analyze_and_store(df_before, "pos1_before", ident)
        analyze_and_store(df_after,  "pos1_after", ident)

for key, recs in results.items():
    if recs:
        out_path = os.path.join(out_dir, f"fit_fallrate_results_{key}.csv")
        pd.DataFrame(recs).to_csv(out_path, index=False)
        print(f"Saved {out_path}")
