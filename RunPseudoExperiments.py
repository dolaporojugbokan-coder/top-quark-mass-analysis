import ROOT
import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

HOME   = "/home/drojugbo"
FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/"
          "Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal")
PARAM  = f"{HOME}/output_parametrisation/Param_mtop.txt"

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
    t  = wdf[0] * mtop + wdf[1]
    t  = np.clip(t, 1e-12, None)
    t  = t / t.sum()
    mu = t * data.sum()
    n  = data
    term = mu - n
    nz = n > 0
    term[nz] += n[nz] * np.log(n[nz] / mu[nz])
    return 2.0 * np.sum(term)

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
    f_tmp = ROOT.TFile.Open(f"{FOLDER}/{filename}")
    h_tmp = f_tmp.Get("h_mtop_param")
    h_tmp.SetDirectory(0)
    f_tmp.Close()
    data_real = np.array([h_tmp.GetBinContent(i) for i in range(1, nbins+1)])

    pulls = []

    for i in range(N_EXPERIMENTS):

        # Step 1+2: Poisson fluctuate → pseudo-data
        data_tmp = np.random.poisson(data_real).astype(float)

        # Step 3: fit
        result = opt.minimize(negLogLik, x0=[m_true], args=(wdf, data_tmp))
        m_meas = result.x[0]

        # Get sigma
        nll_min   = negLogLik(m_meas, wdf, data_tmp)
        mass_scan = np.linspace(m_meas - 0.3, m_meas + 0.3, 10000)
        nll_scan  = np.array([negLogLik(m, wdf, data_tmp) for m in mass_scan])
        below_one = mass_scan[(nll_scan - nll_min) < 1.0]
        sig       = (below_one[-1] - below_one[0]) / 2.0

        # Compute pull and store
        pull = (m_meas - m_true) / sig
        pulls.append(pull)

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{N_EXPERIMENTS} done...")

    pulls = np.array(pulls)

    # ── Plot pull distribution ───────────────────────────────────────────────
    mu_pull, std_pull = norm.fit(pulls)
    print(f"Pull mean = {mu_pull:.3f}, std = {std_pull:.3f}")

    plt.figure(figsize=(7, 5))
    plt.hist(pulls, bins=50, density=True, histtype='step',
         color='steelblue', linewidth=1.5, label='Pull distribution')
    x = np.linspace(pulls.min(), pulls.max(), 300)
    plt.plot(x, norm.pdf(x, mu_pull, std_pull), 'r-', lw=2,
             label=f'Gaussian fit: $\\mu$={mu_pull:.3f}, $\\sigma$={std_pull:.3f}')
    plt.plot(x, norm.pdf(x, 0, 1), 'k--', lw=1.5, label='Ideal: $\\mu$=0, $\\sigma$=1')
    plt.xlabel("Pull")
    plt.ylabel("Density")
    plt.title(f"Pull distribution for $m_{{top}}$ = {m_true} GeV")
    plt.legend()
    plt.tight_layout()
    mass_str = str(m_true).replace('.', 'p')
    plt.savefig(f"{HOME}/output_closure/pull_{mass_str}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pull_{mass_str}.png")

print("\nAll done!")
