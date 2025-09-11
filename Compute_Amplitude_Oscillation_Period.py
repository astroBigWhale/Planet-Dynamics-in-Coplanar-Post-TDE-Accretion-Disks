import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import os
import glob

in_dir = "C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Results"
results_out = "C:/Users/dobru/OneDrive/Bureau/Orbital_movement_of_exoplanets/Treated_eccs"

Ytmin = 365.25 * 24 * 60

dist_files = glob.glob(os.path.join(in_dir, "eccentricities_(*).csv"))

if len(dist_files) == 0:
    print("BOOM!")


for f in dist_files:
    df = pd.read_csv(f)  
    t = df["years"].values
    ecc = df["eccentricity"].values


    peaks, _ = find_peaks(ecc)
    troughs, _ = find_peaks(-ecc)

    records = []


    for i in range(1, len(peaks)):
        peak_prev, peak_curr = peaks[i-1], peaks[i]

    
        period = (t[peak_curr] - t[peak_prev]) * Ytmin

        nearest_trough = min(troughs, key=lambda k: abs(k - peak_curr))
        amplitude = abs(ecc[peak_curr] - ecc[nearest_trough])

        records.append((t[peak_curr], period, amplitude))

    #Save da results
    out_name = os.path.basename(f).replace("eccentricities", "eccentricity_wobble")
    out_df = pd.DataFrame(records, columns=["time [years]", "wobble period [min]", "amplitude"])
    out_path = os.path.join(results_out, out_name)
    out_df.to_csv(out_path, index=False)

    print(f"Saved wobble analysis to {f}")





