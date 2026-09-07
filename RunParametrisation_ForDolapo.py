import os
import ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path to the folder containing the 5 MC ROOT files - new ForDolapo
# standard selection production
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
          "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_NOM_FS/Merged_nominal")

OUTPUT_DIR = "/home/drojugbo/output_parametrisation_ForDolapo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The 5 MC signal files and their corresponding top mass values
file_names = [
    "Merge_Hist_Signal_PP8_171_Comb.root",
    "Merge_Hist_Signal_PP8_172_Comb.root",
    "Merge_Hist_Signal_PP8_Comb.root",
    "Merge_Hist_Signal_PP8_173_Comb.root",
    "Merge_Hist_Signal_PP8_174_Comb.root"
]
mass_values = [171.0, 172.0, 172.5, 173.0, 174.0]

# Read each histogram and normalise to 1 so we compare shapes only
# Different mass points have different event counts so normalisation
# removes that difference before fitting
histograms = []
for files in file_names:
    reading_the_file = ROOT.TFile.Open(f"{FOLDER}/{files}")
    if not reading_the_file or reading_the_file.IsZombie():
        raise RuntimeError(f"Could not open {FOLDER}/{files}")
    hist = reading_the_file.Get("h_mtop_param")
    if not hist:
        raise RuntimeError(f"Could not find h_mtop_param in {FOLDER}/{files}")
    hist.SetDirectory(0)
    reading_the_file.Close()
    integral = hist.Integral()
    if integral <= 0:
        raise ValueError(f"Histogram integral is not positive for {files}")
    hist.Scale(1.0 / integral)
    histograms.append(hist)

nbins      = histograms[0].GetNbinsX()
print(f"Loaded {nbins} bins from {FOLDER}")

slopes     = []
intercepts = []

# For each bin fit a straight line f(m) = slope*m + intercept
# through the 5 normalised bin contents vs mass
# This allows us to predict the bin content at any mass value
for bin_idx in range(1, nbins + 1):
    contents = []
    errors   = []
    for hist in histograms:
        contents.append(hist.GetBinContent(bin_idx))
        errors.append(hist.GetBinError(bin_idx))

    # Plot the 5 points and the fitted line for visual inspection
    plt.errorbar(mass_values, contents, yerr=errors, fmt='o')
    slope, intercept = np.polyfit(mass_values, contents, 1)
    slopes.append(slope)
    intercepts.append(intercept)
    fit_line = [slope * m + intercept for m in mass_values]
    plt.plot(mass_values, fit_line, 'r-')
    plt.title(f"Bin {bin_idx}")
    plt.xlabel("m_top MC [GeV]")
    plt.ylabel("bin content")
    plt.savefig(f"{OUTPUT_DIR}/plot_bin{bin_idx}.png")
    plt.close()

# Save slopes and intercepts for use in the likelihood fit
np.save(f"{OUTPUT_DIR}/slopes.npy", slopes)
np.save(f"{OUTPUT_DIR}/intercepts.npy", intercepts)

with open(f"{OUTPUT_DIR}/Param_mtop.txt", "w") as f:
    for i in range(len(slopes)):
        f.write(f"{i+1},    {slopes[i]},    {intercepts[i]}\n")

print(f"Param_mtop.txt saved to {OUTPUT_DIR}!")
