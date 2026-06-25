import ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = "/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal"

file_names = ["Merge_Hist_Signal_PP8_171_Comb.root",
              "Merge_Hist_Signal_PP8_172_Comb.root",
              "Merge_Hist_Signal_PP8_Comb.root",
              "Merge_Hist_Signal_PP8_173_Comb.root",
              "Merge_Hist_Signal_PP8_174_Comb.root"]

mass_values = [171.0, 172.0, 172.5, 173.0, 174.0]

histograms = []
for files in file_names:
    reading_the_file = ROOT.TFile.Open(f"{folder}/{files}")
    hist = reading_the_file.Get("h_mtop_param")
    hist.SetDirectory(0)
    hist.Scale(1.0 / hist.Integral())
    histograms.append(hist)

nbins = histograms[0].GetNbinsX()
slopes = []
intercepts = []
for bin in range(1, nbins+1):
    contents = []
    errors = [] 
    for hist in histograms:
        contents.append(hist.GetBinContent(bin))
        errors.append(hist.GetBinError(bin))
    plt.errorbar(mass_values, contents, yerr=errors, fmt='o')
    slope, intercept = np.polyfit(mass_values, contents, 1)
    slopes.append(slope)
    intercepts.append(intercept)
    fit_line = [slope * m + intercept for m in mass_values]
    plt.plot(mass_values, fit_line, 'r-')
    plt.title(f"Bin {bin}")
    plt.xlabel("m_top MC [GeV]")
    plt.ylabel("bin content")
    plt.savefig(f"/home/drojugbo/output_parametrisation/plot_bin{bin}.png")
    plt.close()

np.save("/home/drojugbo/output_parametrisation/slopes.npy", slopes)
np.save("/home/drojugbo/output_parametrisation/intercepts.npy", intercepts)
with open("/home/drojugbo/output_parametrisation/Param_mtop.txt", "w") as f:
    for i in range(len(slopes)):
        f.write(f"{i+1},    {slopes[i]},    {intercepts[i]}\n")
print("Param_mtop.txt saved!")
