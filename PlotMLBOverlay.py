import ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME   = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/"
          "Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal")

file_names = [
    "Merge_Hist_Signal_PP8_171_Comb.root",
    "Merge_Hist_Signal_PP8_172_Comb.root",
    "Merge_Hist_Signal_PP8_Comb.root",
    "Merge_Hist_Signal_PP8_173_Comb.root",
    "Merge_Hist_Signal_PP8_174_Comb.root"
]
mass_values = [171.0, 172.0, 172.5, 173.0, 174.0]
colors      = ['blue', 'green', 'red', 'orange', 'purple']

# Read all histograms
all_counts = []
bin_edges  = None

for filename in file_names:
    f    = ROOT.TFile.Open(f"{FOLDER}/{filename}")
    hist = f.Get("h_mtop_param")
    hist.SetDirectory(0)
    f.Close()

    nbins  = hist.GetNbinsX()
    edges  = np.array([hist.GetBinLowEdge(i) for i in range(1, nbins+2)])
    counts = np.array([hist.GetBinContent(i) for i in range(1, nbins+1)], dtype=float)

    if bin_edges is None:
        bin_edges = edges

    all_counts.append(counts / counts.sum())

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
ref_counts  = all_counts[2]

# Two panel plot joined together
fig, (ax_top, ax_ratio) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0}
)

# Join the panels
ax_top.spines['bottom'].set_visible(False)
ax_top.tick_params(bottom=False)
ax_ratio.spines['top'].set_visible(False)

# Top panel: clean step histograms no error bars
for counts, mass, color in zip(all_counts, mass_values, colors):
    lw = 2.5 if mass == 172.5 else 1.5
    ax_top.step(bin_edges[:-1], counts, where='post',
                color=color, linewidth=lw,
                label=f'$m_{{top}}$ = {mass} GeV')

ax_top.set_ylabel("Normalised bin content", fontsize=11)
ax_top.set_title(r"Normalised $m_{\ell b}$ templates", fontsize=12)
ax_top.legend(fontsize=8, loc='upper right')

# Bottom panel: ratio to 172.5 GeV reference
ax_ratio.axhline(1.0, color='red', linestyle='--', linewidth=1.5)

for counts, mass, color in zip(all_counts, mass_values, colors):
    if mass == 172.5:
        continue
    valid = ref_counts > 0
    ratio = np.full_like(counts, np.nan)
    ratio[valid] = counts[valid] / ref_counts[valid]
    ax_ratio.step(bin_edges[:-1], ratio, where='post',
                  color=color, linewidth=1.5)

ax_ratio.set_ylabel("Ratio to\n172.5 GeV", fontsize=10)
ax_ratio.set_xlabel(r"$m_{\ell b}$ [GeV]", fontsize=11)
ax_ratio.set_ylim(0.95, 1.05)

plt.savefig(f"{HOME}/output_parametrisation/mlb_overlay_ratio_clean.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved mlb_overlay_ratio_clean.png")
