import ROOT
import numpy as np

HOME   = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/"
          "Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal")
PARAM  = f"{HOME}/output_parametrisation/Param_mtop.txt"
# Load slopes / intercepts
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

# Build pseudo data = f(172.5)
pseudo_raw  = slopes * 172.5 + intercepts
pseudo_raw  = np.clip(pseudo_raw, 1e-12, None)
pseudo_norm = pseudo_raw / pseudo_raw.sum()
print(f"Pseudo data sum = {pseudo_norm.sum():.6f}")  # should be 1.0
# Get N from the real 172.5 histogram
f_comb = ROOT.TFile.Open(f"{FOLDER}/Merge_Hist_Signal_PP8_Comb.root")
h_comb = f_comb.Get("h_mtop_param")
h_comb.SetDirectory(0)
f_comb.Close()

N = sum(h_comb.GetBinContent(i) for i in range(1, nbins+1))
print(f"N = {N:.0f} events")

# Scale pseudo data to N events
data_counts = pseudo_norm * N
print(f"data_counts sum = {data_counts.sum():.0f}")  # should equal N

# -2lnL function
def template_counts(mtop):
    t = slopes * mtop + intercepts
    t = np.clip(t, 1e-12, None)
    t = t / t.sum()
    return t * N

def minus2lnL(mtop):
    mu   = template_counts(mtop)
    n    = data_counts
    term = mu - n
    nz   = n > 0
    term[nz] += n[nz] * np.log(n[nz] / mu[nz])
    return 2.0 * np.sum(term)

# Calculate -2lnL at the 5 mass points
mass_points = [171.0, 172.0, 172.5, 173.0, 174.0]
nll_points  = [minus2lnL(m) for m in mass_points]

print("\n5 mass points:")
for m, v in zip(mass_points, nll_points):
    print(f"  m = {m:.1f} GeV  ->  -2lnL = {v:.4f}")
# Fit parabola to 5 points
a, b, c = np.polyfit(mass_points, nll_points, 2)
m_hat   = -b / (2.0 * a)          # minimum of parabola
nll_min = np.polyval([a,b,c], m_hat)  # -2lnL at minimum
sigma   = 1.0 / np.sqrt(a)        # statistical uncertainty

print(f"\nMeasured mass : {m_hat:.3f} GeV")
print(f"-2lnL minimum : {nll_min:.4f}")
print(f"Uncertainty   : +/- {sigma:.3f} GeV")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# smooth parabola for plotting
m_scan      = np.linspace(170.5, 174.5, 500)
m_scan_zoom      = np.linspace(m_hat - 5*sigma, m_hat + 5*sigma, 2000)
nll_shifted_zoom = np.polyval([a, b, c], m_scan_zoom) - nll_min
nll_fit     = np.polyval([a, b, c], m_scan)
nll_shifted = nll_fit - nll_min        # shift down so minimum = 0

# Plot 1: -2lnL with 5 points and parabola
plt.figure(figsize=(7, 5))
plt.plot(m_scan, nll_fit, 'r-', lw=2)
plt.plot(mass_points, nll_points, 'kx', ms=8, mew=2, label='5 mass points')
plt.axvline(m_hat, color='grey', ls=':', lw=1)
plt.xlabel(r"$m_{top}$ [GeV]")
plt.ylabel(r"$-2\ln L$")
plt.legend()
plt.tight_layout()
plt.savefig(f"{HOME}/output_likelihood/likelihood_plot1_172p5.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: original + shifted
plt.figure(figsize=(7, 5))
plt.plot(m_scan_zoom, nll_shifted_zoom, 'b-', lw=2, label=r"$-2\Delta\ln L$")
plt.axhline(1.0, color='black', ls='--', lw=1, label=r"$\Delta(-2\ln L) = 1 \rightarrow \pm1\sigma$")
plt.axvline(m_hat, color='grey', ls=':', lw=1,
            label=f"$\\hat{{m}}$ = {m_hat:.3f} $\\pm$ {sigma:.3f} GeV")
plt.ylim(0, 3)
plt.xlim(m_hat - 5*sigma, m_hat + 5*sigma)
plt.xlabel(r"$m_{top}$ [GeV]")
plt.ylabel(r"$-2\Delta\ln L$")
plt.legend()
plt.tight_layout()
plt.savefig(f"{HOME}/output_likelihood/likelihood_plot2_172p5.png", dpi=150, bbox_inches='tight')
plt.close()


# Linearity / bias check: run fit for all 5 pseudo-data masses
# -------------------------------------------------------
mass_list = [171.0, 172.0, 172.5, 173.0, 174.0]
file_list = [
    "Merge_Hist_Signal_PP8_171_Comb.root",
    "Merge_Hist_Signal_PP8_172_Comb.root",
    "Merge_Hist_Signal_PP8_Comb.root",
    "Merge_Hist_Signal_PP8_173_Comb.root",
    "Merge_Hist_Signal_PP8_174_Comb.root"
]

bias_values  = []
bias_errors  = []

for m_true, filename in zip(mass_list, file_list):
    # build pseudo data from parametrized template at m_true
    pd_raw  = slopes * m_true + intercepts
    pd_raw  = np.clip(pd_raw, 1e-12, None)
    pd_norm = pd_raw / pd_raw.sum()

    # get N from the real histogram
    f_tmp = ROOT.TFile.Open(f"{FOLDER}/{filename}")
    h_tmp = f_tmp.Get("h_mtop_param")
    h_tmp.SetDirectory(0)
    f_tmp.Close()
    N_tmp = sum(h_tmp.GetBinContent(i) for i in range(1, nbins+1))
    print(f"m_true = {m_true} | N_tmp = {N_tmp:.0f}")
    data_tmp = pd_norm * N_tmp

    # -2lnL at 5 mass points
    def nll_tmp(mtop, N_tmp, data_tmp):
        t = slopes * mtop + intercepts
        t = np.clip(t, 1e-12, None)
        t = t / t.sum()
        mu = t * N_tmp
        n  = data_tmp
        term = mu - n
        nz = n > 0
        term[nz] += n[nz] * np.log(n[nz] / mu[nz])
        return 2.0 * np.sum(term)
    
    # continuous scan around true mass
    mass_scan    = np.linspace(m_true - 1.5, m_true + 1.5, 1000)
    nll_scan     = np.array([nll_tmp(m, N_tmp, data_tmp) for m in mass_scan])
    index_min    = np.argmin(nll_scan)
    mass_at_min  = mass_scan[index_min]
    near_min     = np.abs(mass_scan - mass_at_min) < 0.5
    mass_near    = mass_scan[near_min]
    nll_near     = nll_scan[near_min]
    aa, bb, cc   = np.polyfit(mass_near, nll_near, 2) 
    m_meas       = -bb / (2.0 * aa)
    sig          = 1.0 / np.sqrt(aa)
    bias = m_true - m_meas
    bias_values.append(bias)
    bias_errors.append(sig)
    print(f"m_true = {m_true:.1f} | m_meas = {m_meas:.3f} | bias = {bias:+.3f} | sigma = {sig:.3f} GeV")
    # smooth parabola for this mass
    m_sc      = np.linspace(170.5, 174.5, 500)
    nll_sc    = np.polyval([aa, bb, cc], m_sc)

    m_sc_zoom      = np.linspace(m_meas - 5*(1/np.sqrt(aa)), m_meas + 5*(1/np.sqrt(aa)), 2000)
    nll_sc_shifted = np.polyval([aa, bb, cc], m_sc_zoom) - np.polyval([aa, bb, cc], m_meas)

    mass_str = str(m_true).replace('.', 'p')

    # Plot 1: raw -2lnL
    plt.figure(figsize=(7, 5))
    plt.plot(m_sc, nll_sc, 'r-', lw=2)
    plt.plot(mass_scan, nll_scan, 'r-', lw=2)   
    plt.axvline(m_meas, color='grey', ls=':', lw=1)
    plt.xlabel(r"$m_{top}$ [GeV]")
    plt.ylabel(r"$-2\ln L$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{HOME}/output_likelihood/likelihood_plot1_{mass_str}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: shifted
    plt.figure(figsize=(7, 5))
    plt.plot(m_sc_zoom, nll_sc_shifted, 'b-', lw=2, label=r"$-2\Delta\ln L$")
    plt.axhline(1.0, color='black', ls='--', lw=1,
                label=r"$\Delta(-2\ln L)=1 \rightarrow \pm1\sigma$")
    plt.axvline(m_meas, color='grey', ls=':', lw=1,
                label=f"$\\hat{{m}}$ = {m_meas:.3f} $\\pm$ {sig:.3f} GeV")
    plt.ylim(0, 3)
    plt.xlabel(r"$m_{top}$ [GeV]")
    plt.ylabel(r"$-2\Delta\ln L$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{HOME}/output_likelihood/likelihood_plot2_{mass_str}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots for m_true = {m_true} GeV")


# weighted mean (fit horizontal line to 5 points)
weights     = [1.0 / (e**2) for e in bias_errors]
mean_bias   = sum(w * b for w, b in zip(weights, bias_values)) / sum(weights)
mean_error  = 1.0 / (sum(weights)**0.5)
print(f"\nFitted bias: {mean_bias:+.3f} +/- {mean_error:.3f} GeV")

# Save results to closure_fit.txt
with open(f"{HOME}/output_likelihood/closure_fit.txt", "w") as out:
    out.write("# m_input   m_measured   uncertainty\n")
    for m, b, e in zip(mass_list, bias_values, bias_errors):
        m_meas = m - b
        out.write(f"{m:.1f}   {m_meas:.6f}   {e:.6f}\n")
print("Saved closure_fit.txt")

# Plot: bias vs m_input with horizontal line at 0
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
plt.savefig(f"{HOME}/output_closure/bias_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved bias_plot.png")
