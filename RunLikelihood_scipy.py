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

# ── Load slopes and intercepts from parametrisation output ───────────────────
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

# ── wdf = parametrisation (slopes and intercepts stacked) ────────────────────
wdf = np.array([slopes, intercepts])

# ── Build pseudo data from f(172.5) ─────────────────────────────────────────
pseudo_raw  = slopes * 172.5 + intercepts
pseudo_raw  = np.clip(pseudo_raw, 1e-12, None)
pseudo_norm = pseudo_raw / pseudo_raw.sum()

# ── Get N from real 172.5 histogram ─────────────────────────────────────────
f_comb = ROOT.TFile.Open(f"{FOLDER}/Merge_Hist_Signal_PP8_Comb.root")
h_comb = f_comb.Get("h_mtop_param")
h_comb.SetDirectory(0)
f_comb.Close()
N = sum(h_comb.GetBinContent(i) for i in range(1, nbins+1))
print(f"N = {N:.0f} events")

# ── Scale pseudo data to N events ────────────────────────────────────────────
data_counts = pseudo_norm * N

# ── Negative log likelihood function ────────────────────────────────────────
def negLogLik(mtop, wdf, data):
    """Baker-Cousins -2lnL between template and pseudo data.
    mtop = top mass to evaluate at
    wdf  = parametrisation [slopes, intercepts]
    data = pseudo data bin counts
    """
    slopes_tmp     = wdf[0]
    intercepts_tmp = wdf[1]
    t    = slopes_tmp * mtop + intercepts_tmp
    t    = np.clip(t, 1e-12, None)
    t    = t / t.sum()
    mu   = t * data.sum()
    n    = data
    term = mu - n
    nz   = n > 0
    term[nz] += n[nz] * np.log(n[nz] / mu[nz])
    return 2.0 * np.sum(term)

# ── Find minimum using scipy for 172.5 ──────────────────────────────────────
result_172p5 = opt.minimize(negLogLik,
                            x0=[172.5],
                            args=(wdf, data_counts))
m_hat = result_172p5.x[0]

# ── Get sigma using profile likelihood scan ──────────────────────────────────
nll_at_min  = negLogLik(m_hat, wdf, data_counts)
mass_scan   = np.linspace(m_hat - 0.3, m_hat + 0.3, 10000)
nll_scan    = np.array([negLogLik(m, wdf, data_counts) for m in mass_scan])
nll_shifted = nll_scan - nll_at_min
below_one   = mass_scan[nll_shifted < 1.0]
sigma       = (below_one[-1] - below_one[0]) / 2.0

print(f"\nMeasured mass : {m_hat:.3f} GeV")
print(f"Uncertainty   : +/- {sigma:.3f} GeV")

# ── Plot 1: raw -2lnL for 172.5 ─────────────────────────────────────────────
mass_plot = np.linspace(170.5, 174.5, 500)
nll_plot  = np.array([negLogLik(m, wdf, data_counts) for m in mass_plot])
plt.figure(figsize=(7, 5))
plt.plot(mass_plot, nll_plot, 'r-', lw=2)
plt.axvline(m_hat, color='grey', ls=':', lw=1)
plt.xlabel(r"$m_{top}$ [GeV]")
plt.ylabel(r"$-2\ln L$")
plt.tight_layout()
plt.savefig(f"{HOME}/output_likelihood/scipy_plot1_172p5.png",
            dpi=150, bbox_inches='tight')
plt.close()

# ── Plot 2: shifted Δ(-2lnL) for 172.5 ──────────────────────────────────────
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
plt.savefig(f"{HOME}/output_likelihood/scipy_plot2_172p5.png",
            dpi=150, bbox_inches='tight')
plt.close()

# ── Bias loop: repeat for all 5 masses using real histogram as pseudo data ───
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

    # ── Read real MC histogram as pseudo data ────────────────────────────────
    f_tmp    = ROOT.TFile.Open(f"{FOLDER}/{filename}")
    h_tmp    = f_tmp.Get("h_mtop_param")
    h_tmp.SetDirectory(0)
    f_tmp.Close()
    N_tmp    = sum(h_tmp.GetBinContent(i) for i in range(1, nbins+1))
    data_tmp = np.array([h_tmp.GetBinContent(i) for i in range(1, nbins+1)])
    data_tmp = np.random.poisson(data_tmp).astype(float)
    print(f"m_true = {m_true} | N_tmp = {N_tmp:.0f}")

    # ── Find minimum using scipy ─────────────────────────────────────────────
    result = opt.minimize(negLogLik,
                          x0=[m_true],
                          args=(wdf, data_tmp))
    m_meas = result.x[0]

    # ── Get sigma using profile likelihood scan ──────────────────────────────
    nll_at_min  = negLogLik(m_meas, wdf, data_tmp)
    mass_scan   = np.linspace(m_meas - 0.3, m_meas + 0.3, 10000)
    nll_scan    = np.array([negLogLik(m, wdf, data_tmp) for m in mass_scan])
    nll_shifted = nll_scan - nll_at_min
    below_one   = mass_scan[nll_shifted < 1.0]
    sig         = (below_one[-1] - below_one[0]) / 2.0

    bias = m_true - m_meas
    bias_values.append(bias)
    bias_errors.append(sig)
    print(f"m_true = {m_true:.1f} | m_meas = {m_meas:.3f} | "
          f"bias = {bias:+.3f} | sigma = {sig:.3f} GeV")

# ── Weighted mean of 5 bias values ───────────────────────────────────────────
weights    = [1.0 / (e**2) for e in bias_errors]
mean_bias  = sum(w * b for w, b in zip(weights, bias_values)) / sum(weights)
mean_error = 1.0 / (sum(weights)**0.5)
print(f"\nFitted bias: {mean_bias:+.3f} +/- {mean_error:.3f} GeV")

# ── Save closure_fit_scipy.txt ───────────────────────────────────────────────
with open(f"{HOME}/output_likelihood/closure_fit_scipy.txt", "w") as out:
    out.write("# m_input   m_measured   uncertainty\n")
    for m, b, e in zip(mass_list, bias_values, bias_errors):
        m_measured = m - b
        out.write(f"{m:.1f}   {m_measured:.6f}   {e:.6f}\n")
print("Saved closure_fit_scipy.txt")

# ── Bias plot ────────────────────────────────────────────────────────────────
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
