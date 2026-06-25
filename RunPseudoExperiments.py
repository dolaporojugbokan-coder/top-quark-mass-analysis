import ROOT
import numpy as np
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME   = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/"
          "Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal")
PARAM  = f"{HOME}/output_parametrisation/Param_mtop.txt"

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
print(f"Loaded {nbins} bins")

# ── Negative log likelihood ──────────────────────────────────────────────────
def negLogLik(mtop, wdf, data):
    template  = wdf[0] * mtop + wdf[1]
    template  = np.clip(template, 1e-12, None)
    template  = template / template.sum()
    expected  = template * data.sum()
    observed  = data
    term      = expected - observed
    nonzero   = observed > 0
    term[nonzero] += observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
    return 2.0 * np.sum(term)

# ── Gaussian function for fitting pull distribution ──────────────────────────
def gaussian(x, mean, width, height):
    return height * np.exp(-0.5 * ((x - mean) / width)**2)

# ── Mass list and files ──────────────────────────────────────────────────────
mass_list = [171.0, 172.0, 172.5, 173.0, 174.0]
file_list = [
    "Merge_Hist_Signal_PP8_171_Comb.root",
    "Merge_Hist_Signal_PP8_172_Comb.root",
    "Merge_Hist_Signal_PP8_Comb.root",
    "Merge_Hist_Signal_PP8_173_Comb.root",
    "Merge_Hist_Signal_PP8_174_Comb.root"
]

N_EXPERIMENTS = 1000

# ── Main loop over all 5 masses ──────────────────────────────────────────────
for m_true, filename in zip(mass_list, file_list):

    print(f"\nRunning {N_EXPERIMENTS} pseudo-experiments for m_true = {m_true} GeV...")

    # Read real MC histogram ONCE
    f_tmp     = ROOT.TFile.Open(f"{FOLDER}/{filename}")
    h_tmp     = f_tmp.Get("h_mtop_param")
    h_tmp.SetDirectory(0)
    f_tmp.Close()
    real_histogram = np.array([h_tmp.GetBinContent(i) for i in range(1, nbins+1)])

    pulls = []

    for i in range(N_EXPERIMENTS):

        # Step 1+2: Poisson fluctuate → pseudo-data
        pseudo_data = np.random.poisson(real_histogram).astype(float)

        # Step 3: fit pseudo data → get measured mass
        fit_result  = opt.minimize(negLogLik, x0=[m_true], args=(wdf, pseudo_data))
        m_measured  = fit_result.x[0]

        # Get sigma from profile likelihood scan
        nll_at_minimum = negLogLik(m_measured, wdf, pseudo_data)
        mass_scan      = np.linspace(m_measured - 0.3, m_measured + 0.3, 10000)
        nll_scan       = np.array([negLogLik(m, wdf, pseudo_data) for m in mass_scan])
        below_one      = mass_scan[(nll_scan - nll_at_minimum) < 1.0]
        sigma_measured = (below_one[-1] - below_one[0]) / 2.0

        # Compute pull and store
        pull = (m_measured - m_true) / sigma_measured
        pulls.append(pull)

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{N_EXPERIMENTS} done...")

    pulls = np.array(pulls)

    # Save pulls to file
    mass_str = str(m_true).replace('.', 'p')
    np.save(f"{HOME}/output_closure/pulls_{mass_str}.npy", pulls)

    # ── Fit Gaussian to pull distribution ───────────────────────────────────
    bin_counts, bin_edges = np.histogram(pulls, bins=50, density=True)
    bin_centers           = (bin_edges[:-1] + bin_edges[1:]) / 2

    best_fit_params, covariance_matrix = curve_fit(gaussian, bin_centers,
                                                    bin_counts, p0=[0, 1, 0.4])
    pull_mean,  pull_width, _  = best_fit_params
    mean_error, width_error, _ = np.sqrt(np.diag(covariance_matrix))

    print(f"Pull mean = {pull_mean:.3f} +/- {mean_error:.3f}, "
          f"std = {pull_width:.3f} +/- {width_error:.3f}")

    # ── Plot pull distribution ───────────────────────────────────────────────
    plt.figure(figsize=(7, 5))
    plt.hist(pulls, bins=50, density=True, histtype='step',
             color='steelblue', linewidth=1.5, label='Pull distribution')
    x_values = np.linspace(pulls.min(), pulls.max(), 300)
    plt.plot(x_values, gaussian(x_values, pull_mean, pull_width, best_fit_params[2]),
             'r-', lw=2,
             label=f'Gaussian fit: $\\mu$={pull_mean:.3f}$\\pm${mean_error:.3f}, '
                   f'$\\sigma$={pull_width:.3f}$\\pm${width_error:.3f}')
    plt.plot(x_values, norm.pdf(x_values, 0, 1), 'k--', lw=1.5,
             label='Ideal: $\\mu$=0, $\\sigma$=1')
    plt.xlabel("Pull")
    plt.ylabel("Density")
    plt.title(f"Pull distribution for $m_{{top}}$ = {m_true} GeV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{HOME}/output_closure/pull_{mass_str}.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pull_{mass_str}.png")

print("\nAll done!")
