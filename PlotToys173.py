import ROOT
import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

HOME   = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/"
          "Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal")
PARAM  = f"{HOME}/output_parametrisation/Param_mtop.txt"
os.makedirs(f"{HOME}/output_closure/toys_173", exist_ok=True)

np.random.seed(42)

# ── Load slopes and intercepts ───────────────────────────────────────────────
slopes, intercepts = [], []
with open(PARAM) as fp:
    for line in fp:
        parts = line.strip().split(",")
        if len(parts) < 3:
            continue
        try:
            slopes.append(float(parts[1]))
            intercepts.append(float(parts[2]))
        except ValueError:
            continue
slopes     = np.array(slopes)
intercepts = np.array(intercepts)
nbins      = len(slopes)
wdf        = np.array([slopes, intercepts])

# ── Negative log likelihood ──────────────────────────────────────────────────
def negLogLik(mtop, wdf, data):
    m        = float(np.atleast_1d(mtop)[0])
    template = wdf[0] * m + wdf[1]
    template = np.clip(template, 1e-12, None)
    template = template / template.sum()
    expected = template * data.sum()
    observed = data
    term     = expected - observed
    nonzero  = observed > 0
    term[nonzero] += observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
    return 2.0 * np.sum(term)

# ── Read real MC histogram for 173.0 ────────────────────────────────────────
m_true    = 173.0
root_file = ROOT.TFile.Open(f"{FOLDER}/Merge_Hist_Signal_PP8_173_Comb.root")
h_tmp     = root_file.Get("h_mtop_param")
h_tmp.SetDirectory(0)
root_file.Close()

real_counts = np.array([h_tmp.GetBinContent(i) for i in range(1, nbins+1)], dtype=float)
real_errors = np.sqrt(real_counts)
bin_edges   = np.array([h_tmp.GetBinLowEdge(i) for i in range(1, nbins+2)], dtype=float)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ── Generate and plot 20 individual pseudo-histograms ────────────────────────
N_PLOTS = 20

for i in range(N_PLOTS):

    # Step 1+2: Poisson fluctuate
    pseudo_data   = np.random.poisson(real_counts).astype(float)
    pseudo_errors = np.sqrt(pseudo_data)

    # Step 3: fit → get measured mass
    fit_result = opt.minimize(negLogLik, x0=[m_true], args=(wdf, pseudo_data))
    m_measured = fit_result.x[0]

    # Get sigma from profile likelihood scan
    nll_at_minimum = negLogLik(m_measured, wdf, pseudo_data)
    mass_scan      = np.linspace(m_measured - 0.3, m_measured + 0.3, 10000)
    nll_scan       = np.array([negLogLik(m, wdf, pseudo_data) for m in mass_scan])
    below_one      = mass_scan[(nll_scan - nll_at_minimum) < 1.0]
    sigma_measured = (below_one[-1] - below_one[0]) / 2.0

    # Ratio and propagated uncertainty
    valid     = (real_counts > 0) & (pseudo_data > 0)
    ratio     = np.full_like(real_counts, np.nan)
    ratio_err = np.full_like(real_counts, np.nan)
    ratio[valid]     = pseudo_data[valid] / real_counts[valid]
    ratio_err[valid] = ratio[valid] * np.sqrt(
        (pseudo_errors[valid] / pseudo_data[valid])**2 +
        (real_errors[valid]   / real_counts[valid])**2
    )

    # ── Two panel plot joined together ───────────────────────────────────────
    fig, (ax_top, ax_ratio) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.0}
    )

    # Join the panels
    ax_top.spines['bottom'].set_visible(False)
    ax_top.tick_params(bottom=False)
    ax_ratio.spines['top'].set_visible(False)

    # Top panel: real histogram (black) + pseudo-data (red) with uncertainties
    ax_top.step(bin_edges[:-1], real_counts, where='post',
                color='black', linewidth=1.5, label='Real MC histogram')
    ax_top.errorbar(bin_centers, real_counts, yerr=real_errors,
                    fmt='none', ecolor='black', elinewidth=1.0, capsize=2)
    ax_top.errorbar(bin_centers, pseudo_data, yerr=pseudo_errors,
                    fmt='o', color='red', markersize=3, elinewidth=1.0,
                    capsize=2, label='Pseudo-data')
    ax_top.set_ylabel("Events")
    ax_top.set_title(f"Pseudo-histogram {i+1}/{N_PLOTS} "
                     f"(from real MC, m_true=173.0)  |  "
                     f"$m_{{top}}^{{fit}}$ = {m_measured:.3f} "
                     f"$\\pm$ {sigma_measured:.3f} GeV")
    ax_top.legend(loc='upper right', fontsize=9)

    # Bottom panel: ratio with uncertainties
    ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1.2)
    ax_ratio.errorbar(bin_centers[valid], ratio[valid], yerr=ratio_err[valid],
                      fmt='o', color='red', markersize=3,
                      elinewidth=1.0, capsize=2)
    ax_ratio.set_ylabel("Pseudo / Real")
    ax_ratio.set_xlabel(r"$m_{lb}$ [GeV]")
    ax_ratio.set_ylim(0.95, 1.05)

    plt.savefig(f"{HOME}/output_closure/toys_173/pseudo_hist_173_{i+1:03d}.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pseudo_hist_173_{i+1:03d}.png  "
          f"(m_fit = {m_measured:.3f} +/- {sigma_measured:.3f} GeV)")

print("\nDone!")
