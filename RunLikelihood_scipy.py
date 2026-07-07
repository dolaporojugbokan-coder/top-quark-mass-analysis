import ROOT
import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME   = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/"
          "Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal")
PARAM  = f"{HOME}/output_parametrisation/Param_mtop.txt"

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
print(f"Loaded {nbins} bins")

# wdf holds the parametrisation: wdf[0]=slopes, wdf[1]=intercepts
wdf = np.array([slopes, intercepts])

# Build pseudo data from the parametrised template at 172.1 GeV
# 172.1 is between the training points so it tests interpolation
pseudo_raw  = slopes * 172.1 + intercepts
pseudo_raw  = np.clip(pseudo_raw, 1e-12, None)
pseudo_norm = pseudo_raw / pseudo_raw.sum()

# Get total events N from the real 172.5 histogram
# to scale the pseudo data to a realistic event count
f_comb = ROOT.TFile.Open(f"{FOLDER}/Merge_Hist_Signal_PP8_Comb.root")
h_comb = f_comb.Get("h_mtop_param")
h_comb.SetDirectory(0)
f_comb.Close()
N = sum(h_comb.GetBinContent(i) for i in range(1, nbins+1))
print(f"N = {N:.0f} events")

# Scale the normalised pseudo data to N events
data_counts = pseudo_norm * N

# Baker-Cousins -2lnL between template and data
def negLogLik(mtop, wdf, data):
    slopes_tmp     = wdf[0]
    intercepts_tmp = wdf[1]
    # Build template at this mass and normalise to shape
    t    = slopes_tmp * mtop + intercepts_tmp
    t    = np.clip(t, 1e-12, None)
    t    = t / t.sum()
    # Scale template to match total events in data
    mu   = t * data.sum()
    n    = data
    term = mu - n
    nz   = n > 0
    term[nz] += n[nz] * np.log(n[nz] / mu[nz])
    return 2.0 * np.sum(term)

def fit_mass(data, lower=169.0, upper=176.0):
    # Find the mass that minimises -2lnL using bounded scalar minimisation
    result = opt.minimize_scalar(
        lambda m: negLogLik(m, wdf, data),
        bounds=(lower, upper),
        method='bounded'
    )
    m_hat   = result.x
    nll_min = negLogLik(m_hat, wdf, data)

    # Find the exact +/-1 sigma crossing points using brentq
    # These are where delta(-2lnL) = 1
    def delta_nll_minus_one(m):
        return negLogLik(m, wdf, data) - nll_min - 1.0

    left_crossing  = opt.brentq(delta_nll_minus_one, lower, m_hat)
    right_crossing = opt.brentq(delta_nll_minus_one, m_hat, upper)
    sigma = 0.5 * (right_crossing - left_crossing)

    return m_hat, sigma, nll_min

# Fit the pseudo data at 172.1 to verify the method recovers the correct mass
m_hat, sigma, nll_at_min = fit_mass(data_counts)

print(f"\nMeasured mass : {m_hat:.3f} GeV")
print(f"Uncertainty   : +/- {sigma:.3f} GeV")

# Plot 1: raw -2lnL curve showing the likelihood minimum
mass_plot = np.linspace(170.5, 174.5, 500)
nll_plot  = np.array([negLogLik(m, wdf, data_counts) for m in mass_plot])
plt.figure(figsize=(7, 5))
plt.plot(mass_plot, nll_plot, 'r-', lw=2)
plt.axvline(m_hat, color='grey', ls=':', lw=1)
plt.xlabel(r"$m_{top}$ [GeV]")
plt.ylabel(r"$-2\ln L$")
plt.tight_layout()
plt.savefig(f"{HOME}/output_likelihood/scipy_plot1_172p1.png",
            dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: shifted delta(-2lnL) showing the +/-1 sigma uncertainty region
mass_zoom        = np.linspace(m_hat - 5*sigma, m_hat + 5*sigma, 2000)
nll_zoom         = np.array([negLogLik(m, wdf, data_counts) for m in mass_zoom])
nll_zoom_shifted = nll_zoom - nll_at_min
plt.figure(figsize=(7, 5))
plt.plot(mass_zoom, nll_zoom_shifted, 'b-', lw=2, label=r"$-2\Delta\ln L$")
plt.axhline(1.0, color='black', ls='--', lw=1,
            label=r"$\Delta(-2\ln L) = 1 \rightarrow \pm1\sigma$")
plt.axvline(m_hat, color='grey', ls=':', lw=1,
            label=f"$\\hat{{m}}$ = {m_hat:.3f} $\\pm$ {sigma:.3f} GeV")
plt.ylim(0, 3)
plt.xlabel(r"$m_{top}$ [GeV]")
plt.ylabel(r"$-2\Delta\ln L$")
plt.legend()
plt.tight_layout()
plt.savefig(f"{HOME}/output_likelihood/scipy_plot2_172p1.png",
            dpi=150, bbox_inches='tight')
plt.close()

# Bias loop: fit each of the 5 real MC histograms to check closure
mass_list = [171.0, 172.0, 172.5, 173.0, 174.0]
file_list = [
    "Merge_Hist_Signal_PP8_171_Comb.root",
    "Merge_Hist_Signal_PP8_172_Comb.root",
    "Merge_Hist_Signal_PP8_Comb.root",
    "Merge_Hist_Signal_PP8_173_Comb.root",
    "Merge_Hist_Signal_PP8_174_Comb.root"
]

bias_values = []
bias_errors = []

for m_true, filename in zip(mass_list, file_list):

    # Read the real MC histogram directly as pseudo data
    f_tmp    = ROOT.TFile.Open(f"{FOLDER}/{filename}")
    h_tmp    = f_tmp.Get("h_mtop_param")
    h_tmp.SetDirectory(0)
    f_tmp.Close()
    N_tmp    = sum(h_tmp.GetBinContent(i) for i in range(1, nbins+1))
    data_tmp = np.array([h_tmp.GetBinContent(i) for i in range(1, nbins+1)])
    print(f"m_true = {m_true} | N_tmp = {N_tmp:.0f}")

    # Fit this histogram to get measured mass and uncertainty
    m_meas, sig, _ = fit_mass(data_tmp)

    # bias = input mass - fitted mass
    bias = m_true - m_meas
    bias_values.append(bias)
    bias_errors.append(sig)
    print(f"m_true = {m_true:.1f} | m_meas = {m_meas:.3f} | "
          f"bias = {bias:+.3f} | sigma = {sig:.3f} GeV")

# Weighted mean bias across all 5 mass points
weights    = [1.0 / (e**2) for e in bias_errors]
mean_bias  = sum(w * b for w, b in zip(weights, bias_values)) / sum(weights)
mean_error = 1.0 / (sum(weights)**0.5)
print(f"\nFitted bias: {mean_bias:+.3f} +/- {mean_error:.3f} GeV")

# Save results for use by RunClosureTest.py
with open(f"{HOME}/output_likelihood/closure_fit_scipy.txt", "w") as out:
    out.write("# m_input   m_measured   uncertainty\n")
    for m, b, e in zip(mass_list, bias_values, bias_errors):
        m_measured = m - b
        out.write(f"{m:.1f}   {m_measured:.6f}   {e:.6f}\n")
print("Saved closure_fit_scipy.txt")

# Plot bias vs input mass
plt.figure(figsize=(7, 5))
plt.errorbar(mass_list, bias_values, yerr=bias_errors,
             fmt='o', color='b', elinewidth=1.5, capsize=4, ms=6)
plt.axhline(mean_bias, color='r', ls='--', lw=1.5,
            label=f'fit: {mean_bias:+.3f} $\\pm$ {mean_error:.3f} GeV')
plt.axhline(0, color='black', ls=':', lw=1)
plt.xlabel(r"$m_{top}^{input}$ [GeV]")
plt.ylabel(r"$m_{top}^{input} - m_{top}^{fit}$ [GeV]")
plt.legend()
plt.tight_layout()
plt.savefig(f"{HOME}/output_closure/bias_plot_scipy.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved bias_plot_scipy.png")
