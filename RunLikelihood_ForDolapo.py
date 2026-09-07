import os
import ROOT
import numpy as np
import scipy.optimize as opt

HOME = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
          "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_NOM_FS/Merged_nominal")
PARAM = f"{HOME}/output_parametrisation_ForDolapo/Param_mtop.txt"
OUTPUT_DIR = f"{HOME}/output_likelihood_ForDolapo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HISTOGRAM_NAME = "h_mtop_param"
LOWER_BOUND = 168.0
UPPER_BOUND = 177.0

mass_files = {
    171.0: "Merge_Hist_Signal_PP8_171_Comb.root",
    172.0: "Merge_Hist_Signal_PP8_172_Comb.root",
    172.5: "Merge_Hist_Signal_PP8_Comb.root",
    173.0: "Merge_Hist_Signal_PP8_173_Comb.root",
    174.0: "Merge_Hist_Signal_PP8_174_Comb.root",
}

# load the new parametrisation
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

slopes = np.array(slopes)
intercepts = np.array(intercepts)
nbins = len(slopes)
wdf = np.array([slopes, intercepts])
print(f"Loaded {nbins} bins from {PARAM}")


def negLogLik(mtop, wdf, data):
    mass = float(np.atleast_1d(mtop)[0])
    template = wdf[0] * mass + wdf[1]
    template = np.clip(template, 1e-12, None)
    template = template / template.sum()
    expected = template * data.sum()
    observed = data
    expected = np.clip(expected, 1e-12, None)
    term = expected - observed
    nonzero = observed > 0
    term[nonzero] += observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
    return 2.0 * np.sum(term)


def read_histogram_counts(filepath, histname, expected_bins):
    root_file = ROOT.TFile.Open(filepath)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open {filepath}")

    histogram = root_file.Get(histname)
    if not histogram:
        root_file.Close()
        raise RuntimeError(f"Could not find {histname} in {filepath}")

    histogram.SetDirectory(0)
    root_file.Close()

    if histogram.GetNbinsX() != expected_bins:
        raise RuntimeError(
            f"Histogram has {histogram.GetNbinsX()} bins, "
            f"but parametrisation has {expected_bins} bins: {filepath}"
        )

    return np.array(
        [histogram.GetBinContent(i) for i in range(1, expected_bins + 1)]
    )


def fit_mass(data):
    result = opt.minimize_scalar(
        lambda mass: negLogLik(mass, wdf, data),
        bounds=(LOWER_BOUND, UPPER_BOUND),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"Fit did not converge: {result.message}")

    m_hat = float(result.x)
    nll_min = negLogLik(m_hat, wdf, data)

    def delta_nll_minus_one(mass):
        return negLogLik(mass, wdf, data) - nll_min - 1.0

    left = opt.brentq(delta_nll_minus_one, LOWER_BOUND, m_hat)
    right = opt.brentq(delta_nll_minus_one, m_hat, UPPER_BOUND)

    sigma_lower = m_hat - left
    sigma_upper = right - m_hat
    return m_hat, sigma_lower, sigma_upper


# fit the nominal 172.5 GeV file directly - this is the headline number
nominal_path = os.path.join(FOLDER, mass_files[172.5])
nominal_counts = read_histogram_counts(nominal_path, HISTOGRAM_NAME, nbins)
N = nominal_counts.sum()
print(f"\nNominal events: {N:.0f}")

m_hat, sigma_lower, sigma_upper = fit_mass(nominal_counts)
print(f"Fitted mass   : {m_hat:.6f} GeV")
print(f"Lower error   : -{sigma_lower:.6f} GeV")
print(f"Upper error   : +{sigma_upper:.6f} GeV")

with open(f"{OUTPUT_DIR}/nominal_fit_result.txt", "w") as out:
    out.write(f"fitted_mass_GeV: {m_hat:.6f}\n")
    out.write(f"sigma_lower_GeV: {sigma_lower:.6f}\n")
    out.write(f"sigma_upper_GeV: {sigma_upper:.6f}\n")
    out.write(f"n_events: {N:.0f}\n")

# closure loop across all 5 mass points
print("\nClosure loop:")
closure_results = []
for m_true, filename in mass_files.items():
    filepath = os.path.join(FOLDER, filename)
    counts = read_histogram_counts(filepath, HISTOGRAM_NAME, nbins)
    n_events = counts.sum()

    try:
        m_meas, sig_lo, sig_hi = fit_mass(counts)
        bias = m_meas - m_true
        sigma_sym = 0.5 * (sig_lo + sig_hi)
        closure_results.append((m_true, m_meas, bias, sigma_sym))
        print(f"m_true = {m_true:.1f} | m_meas = {m_meas:.3f} | "
              f"bias = {bias:+.3f} | sigma = {sigma_sym:.3f} GeV | N={n_events:.0f}")
    except RuntimeError as error:
        print(f"m_true = {m_true:.1f} | FAILED - {error}")

with open(f"{OUTPUT_DIR}/closure_fit_ForDolapo.txt", "w") as out:
    out.write("# m_true, m_meas, bias, sigma\n")
    for m_true, m_meas, bias, sigma in closure_results:
        out.write(f"{m_true:.6f}, {m_meas:.6f}, {bias:+.6f}, {sigma:.6f}\n")

if closure_results:
    biases = np.array([r[2] for r in closure_results])
    print(f"\nLargest |bias|: {np.max(np.abs(biases)):.4f} GeV")

print(f"\nSaved results to {OUTPUT_DIR}")
