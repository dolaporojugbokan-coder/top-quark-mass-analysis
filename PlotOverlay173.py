import ROOT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

HOME   = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/"
          "Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal")
os.makedirs(f"{HOME}/output_closure", exist_ok=True)

np.random.seed(42)

# ── Choose the mass point ────────────────────────────────────────────────────
m_true         = 173.0
filename       = "Merge_Hist_Signal_PP8_173_Comb.root"
histname       = "h_mtop_param"
N_TOYS_TO_PLOT = 20

# ── Read original ROOT histogram ─────────────────────────────────────────────
f = ROOT.TFile.Open(f"{FOLDER}/{filename}")
h = f.Get(histname)
h.SetDirectory(0)
f.Close()

nbins        = h.GetNbinsX()
bin_contents = np.array([h.GetBinContent(i) for i in range(1, nbins+1)], dtype=float)
bin_errors   = np.sqrt(bin_contents)
bin_edges    = np.array([h.GetBinLowEdge(i) for i in range(1, nbins+2)], dtype=float)
bin_centers  = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ── Generate 20 toy histograms ───────────────────────────────────────────────
toy_histograms = []
toy_errors_all = []

for i in range(N_TOYS_TO_PLOT):
    toy     = np.random.poisson(bin_contents).astype(float)
    toy_err = np.sqrt(toy)
    toy_histograms.append(toy)
    toy_errors_all.append(toy_err)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, (ax_top, ax_ratio) = plt.subplots(
    2, 1, figsize=(9, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.0}
)

# Remove borders between panels so they look joined
ax_top.spines['bottom'].set_visible(False)
ax_top.tick_params(bottom=False)
ax_ratio.spines['top'].set_visible(False)

colors = plt.cm.tab20(np.linspace(0, 1, N_TOYS_TO_PLOT))

# ── Top panel ────────────────────────────────────────────────────────────────
# Original histogram
ax_top.step(bin_edges[:-1], bin_contents, where='post',
            color='black', linewidth=2.0, label='Original histogram')
ax_top.errorbar(bin_centers, bin_contents, yerr=bin_errors,
                fmt='none', ecolor='black', elinewidth=1.0, capsize=2)

# 20 toys
for i, (toy, toy_err) in enumerate(zip(toy_histograms, toy_errors_all)):
    ax_top.step(bin_edges[:-1], toy, where='post',
                linewidth=1.0, alpha=0.7, color=colors[i],
                label=f'Toy {i+1}' if i < 5 else None)
    ax_top.errorbar(bin_centers, toy, yerr=toy_err,
                    fmt='none', elinewidth=0.6, capsize=1,
                    alpha=0.4, ecolor=colors[i])

ax_top.set_ylabel("Events")
ax_top.set_title(f"Original histogram and first {N_TOYS_TO_PLOT} toys ({m_true} GeV)")
ax_top.legend(loc='upper right', fontsize=8)

# ── Bottom panel: ratio ───────────────────────────────────────────────────────
ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1.2)

for i, (toy, toy_err) in enumerate(zip(toy_histograms, toy_errors_all)):
    ratio     = np.full_like(bin_contents, np.nan, dtype=float)
    ratio_err = np.full_like(bin_contents, np.nan, dtype=float)
    valid     = (bin_contents > 0) & (toy > 0)
    ratio[valid]     = toy[valid] / bin_contents[valid]
    ratio_err[valid] = ratio[valid] * np.sqrt(
        (toy_err[valid] / toy[valid])**2 +
        (bin_errors[valid] / bin_contents[valid])**2
    )
    ax_ratio.step(bin_edges[:-1], ratio, where='post',
                  linewidth=1.0, alpha=0.7, color=colors[i])
    ax_ratio.errorbar(bin_centers[valid], ratio[valid], yerr=ratio_err[valid],
                      fmt='none', elinewidth=0.6, capsize=1,
                      alpha=0.4, ecolor=colors[i])

ax_ratio.set_ylabel("Toy / Orig.")
ax_ratio.set_xlabel(r"$m_{lb}$ [GeV]")
ax_ratio.set_ylim(0.95, 1.05)

plt.savefig(f"{HOME}/output_closure/original_plus_20toys_ratio_173p0_with_errors.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved original_plus_20toys_ratio_173p0_with_errors.png")
