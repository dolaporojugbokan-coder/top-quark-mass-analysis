import os
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
OUTPUT_DIR = f"{HOME}/output_closure"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fix random seed for reproducibility
np.random.seed(42)

# Load per-bin slopes and intercepts saved by RunParametrisation.py
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

def negLogLik(mtop, wdf, data):
    m         = float(np.atleast_1d(mtop)[0])
    template  = wdf[0] * m + wdf[1]
    template  = np.clip(template, 1e-12, None)
    template  = template / template.sum()
    expected  = template * data.sum()
    observed  = data
    term      = expected - observed
    nonzero   = observed > 0
    term[nonzero] += observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
    return 2.0 * np.sum(term)

def gaussian(x, mean, width, height):
    return height * np.exp(-0.5 * ((x - mean) / width)**2)

mass_list = [171.0, 172.0, 172.5, 173.0, 174.0]
file_list = [
    "Merge_Hist_Signal_PP8_171_Comb.root",
    "Merge_Hist_Signal_PP8_172_Comb.root",
    "Merge_Hist_Signal_PP8_Comb.root",
    "Merge_Hist_Signal_PP8_173_Comb.root",
    "Merge_Hist_Signal_PP8_174_Comb.root"
]

N_EXPERIMENTS = 100000

# fixed seed and bounds used for every fit, for every mass point
SEED_MASS   = 169.5
LOWER_BOUND = 168.0
UPPER_BOUND = 177.0

for m_true, filename in zip(mass_list, file_list):

    print(f"\nRunning {N_EXPERIMENTS} pseudo-experiments for m_true = {m_true} GeV...")

    f_tmp     = ROOT.TFile.Open(f"{FOLDER}/{filename}")
    h_tmp     = f_tmp.Get("h_mtop_param")
    h_tmp.SetDirectory(0)
    f_tmp.Close()
    real_histogram = np.array([h_tmp.GetBinContent(i) for i in range(1, nbins+1)])

    pulls           = []
    m_measured_list = []
    upper_errors    = []
    lower_errors    = []
    failed_fits     = 0

    for i in range(N_EXPERIMENTS):

        pseudo_data = np.random.poisson(real_histogram).astype(float)

        # seed fixed at 169.5 (not m_true) so the fit has to find its way
        # to the answer from far away, rather than starting on top of it
        fit_result = opt.minimize(
            negLogLik,
            x0=[SEED_MASS],
            args=(wdf, pseudo_data),
            method="L-BFGS-B",
            bounds=[(LOWER_BOUND, UPPER_BOUND)],
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000}
        )
        m_measured = fit_result.x[0]

        if not fit_result.success:
            failed_fits += 1

        nll_at_minimum = negLogLik(m_measured, wdf, pseudo_data)

        def delta_nll_minus_one(m):
            return negLogLik(m, wdf, pseudo_data) - nll_at_minimum - 1.0

        try:
            left_crossing  = opt.brentq(delta_nll_minus_one, LOWER_BOUND, m_measured)
            right_crossing = opt.brentq(delta_nll_minus_one, m_measured, UPPER_BOUND)
        except ValueError:
            failed_fits += 1
            continue

        sigma_measured = 0.5 * (right_crossing - left_crossing)

        pull = (m_measured - m_true) / sigma_measured
        pulls.append(pull)

        m_measured_list.append(m_measured)
        upper_errors.append(right_crossing - m_measured)
        lower_errors.append(m_measured - left_crossing)

        if (i+1) % 5000 == 0:
            print(f"  {i+1}/{N_EXPERIMENTS} done...")

    print(f"  Failed fits: {failed_fits}/{N_EXPERIMENTS}")

    pulls          = np.array(pulls)
    m_measured_arr = np.array(m_measured_list)
    upper_arr      = np.array(upper_errors)
    lower_arr      = np.array(lower_errors)
    residuals      = m_measured_arr - m_true
    mass_str       = str(m_true).replace('.', 'p')

    residual_mean       = np.mean(residuals)
    residual_std        = np.std(residuals, ddof=1)
    residual_mean_error = residual_std / np.sqrt(len(residuals))

    print(f"Residual mean = {residual_mean:+.6f} +/- {residual_mean_error:.6f} GeV")

    with open(f"{OUTPUT_DIR}/residual_summary_{mass_str}_100k.txt", "w") as out:
        out.write("# m_true   mean_residual   error_on_mean   residual_std\n")
        out.write(f"{m_true:.1f}   {residual_mean:.9f}   "
                  f"{residual_mean_error:.9f}   {residual_std:.9f}\n")
    print(f"Saved residual_summary_{mass_str}_100k.txt")

    np.save(f"{OUTPUT_DIR}/pulls_{mass_str}_100k.npy", pulls)

    bin_counts, bin_edges = np.histogram(pulls, bins=100, density=True)
    bin_centers           = (bin_edges[:-1] + bin_edges[1:]) / 2

    best_fit_params, covariance_matrix = curve_fit(gaussian, bin_centers,
                                                    bin_counts, p0=[0, 1, 0.4])
    pull_mean,  pull_width, _  = best_fit_params
    mean_error, width_error, _ = np.sqrt(np.diag(covariance_matrix))

    print(f"Pull mean = {pull_mean:.3f} +/- {mean_error:.3f}, "
          f"std = {pull_width:.3f} +/- {width_error:.3f}")

    plt.figure(figsize=(7, 5))
    plt.hist(pulls, bins=100, density=True, histtype='step',
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
    plt.title(f"Pull distribution for $m_{{top}}$ = {m_true} GeV (100k experiments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pull_{mass_str}_100k.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pull_{mass_str}_100k.png")

    plt.figure(figsize=(7, 5))
    plt.hist(m_measured_arr, bins=100, histtype='step',
             color='steelblue', linewidth=1.5)
    plt.axvline(m_true, color='red', ls='--', lw=1.5,
                label=f'True mass = {m_true} GeV')
    plt.axvline(m_measured_arr.mean(), color='black', ls=':', lw=1.5,
                label=f'Mean = {m_measured_arr.mean():.4f} GeV')
    plt.xlabel(r"$m_{top}^{fit}$ [GeV]")
    plt.ylabel("Experiments")
    plt.title(f"Fitted $m_{{top}}$ distribution for {m_true} GeV (100k experiments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/m_measured_{mass_str}_100k.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved m_measured_{mass_str}_100k.png")

    counts, bin_edges_res = np.histogram(residuals, bins=100)
    bin_centers_res       = (bin_edges_res[:-1] + bin_edges_res[1:]) / 2
    bin_errors_res        = np.sqrt(counts)

    plt.figure(figsize=(7, 5))
    plt.errorbar(bin_centers_res, counts, yerr=bin_errors_res,
                 fmt='o', color='steelblue', markersize=3,
                 elinewidth=1.0, capsize=2)
    plt.axvline(0, color='red', ls='--', lw=1.5, label='Zero')
    plt.axvline(residual_mean, color='black', ls=':', lw=1.5,
                label=f'Mean = {residual_mean:+.6f} $\\pm$ {residual_mean_error:.6f} GeV')
    plt.xlabel(r"$m_{top}^{fit} - m_{top}^{true}$ [GeV]")
    plt.ylabel("Experiments")
    plt.title(f"Residual $m_{{top}}^{{fit}} - m_{{top}}^{{true}}$ for {m_true} GeV (100k)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/residual_{mass_str}_100k.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved residual_{mass_str}_100k.png")

    plt.figure(figsize=(7, 5))
    plt.hist(upper_arr, bins=100, histtype='step',
             color='steelblue', linewidth=1.5)
    plt.axvline(upper_arr.mean(), color='red', ls='--', lw=1.5,
                label=f'Mean = {upper_arr.mean():.4f} GeV')
    plt.xlabel(r"Upper error $\sigma^{+}$ [GeV]")
    plt.ylabel("Experiments")
    plt.title(f"Upper error distribution for {m_true} GeV (100k experiments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/upper_error_{mass_str}_100k.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved upper_error_{mass_str}_100k.png")

    plt.figure(figsize=(7, 5))
    plt.hist(lower_arr, bins=100, histtype='step',
             color='steelblue', linewidth=1.5)
    plt.axvline(lower_arr.mean(), color='red', ls='--', lw=1.5,
                label=f'Mean = {lower_arr.mean():.4f} GeV')
    plt.xlabel(r"Lower error $\sigma^{-}$ [GeV]")
    plt.ylabel("Experiments")
    plt.title(f"Lower error distribution for {m_true} GeV (100k experiments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/lower_error_{mass_str}_100k.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved lower_error_{mass_str}_100k.png")

print("\nAll done!")
