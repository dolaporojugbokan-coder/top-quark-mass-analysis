import os
import ROOT
import numpy as np
import scipy.optimize as opt

HOME = "/home/drojugbo"
PARAM_FILE = f"{HOME}/output_parametrisation_ForDolapo/Param_mtop.txt"

SYS_FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
              "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_SYSLJ_FS")
BASE_AF = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
           "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_NOM_AF/Merged_nominal")

HIST_NAME = "h_mtop_param"
LOWER_BOUND = 168.0
UPPER_BOUND = 177.0

slopes, intercepts = [], []
with open(PARAM_FILE) as file:
    for line in file:
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
print(f"Loaded {nbins} bins")


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


def fit_histogram(root_path):
    root_file = ROOT.TFile.Open(root_path)
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open {root_path}")
    histogram = root_file.Get(HIST_NAME)
    if not histogram:
        root_file.Close()
        raise RuntimeError(f"Could not find {HIST_NAME} in {root_path}")
    histogram.SetDirectory(0)
    root_file.Close()
    if histogram.GetNbinsX() != nbins:
        raise RuntimeError(f"Bin mismatch: {root_path}")
    data = np.array([histogram.GetBinContent(i) for i in range(1, nbins + 1)])
    if data.sum() <= 0:
        raise RuntimeError(f"Non-positive integral: {root_path}")
    fit_result = opt.minimize_scalar(
        lambda mass: negLogLik(mass, wdf, data),
        bounds=(LOWER_BOUND, UPPER_BOUND), method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )
    if not fit_result.success:
        raise RuntimeError(f"Fit failed: {root_path}")
    return fit_result.x, data.sum()


def calc_up_down_systematic(up, down, nominal):
    lower, upper = min(up, down), max(up, down)
    if lower <= nominal <= upper:
        return abs(up - down) / 2.0, "half difference"
    delta_1 = up - nominal
    delta_2 = down - nominal
    if abs(delta_1) > abs(delta_2):
        return abs(delta_1), "largest shift (up)"
    return abs(delta_2), "largest shift (down)"


# LineShape: one-sided comparison against nominal AF
nominal_af_path = f"{BASE_AF}/Merge_Hist_Signal_PP8_Comb.root"
lineshape_path = f"{BASE_AF}/Merge_Hist_Signal_LineShape_Comb.root"

nominal_af_mass, nominal_af_n = fit_histogram(nominal_af_path)
lineshape_mass, lineshape_n = fit_histogram(lineshape_path)

lineshape_unc = abs(lineshape_mass - nominal_af_mass)

print(f"\nLineShape")
print(f"  nominal AF = {nominal_af_mass:.6f} GeV")
print(f"  LineShape  = {lineshape_mass:.6f} GeV")
print(f"  uncertainty = {lineshape_unc:.6f} GeV")

# Pileup: up/down pair, compared against the SYSLJ nominal
# (same nominal used for all other Out_SYSLJ_FS systematics)
nominal_sys_path = os.path.join(SYS_FOLDER, "Merged_nominal", "Merge_Hist_Signal_PP8_Comb.root")
pileup_up_path = os.path.join(SYS_FOLDER, "Merged_weight_pileup_UP", "Merge_Hist_Signal_PP8_Comb.root")
pileup_down_path = os.path.join(SYS_FOLDER, "Merged_weight_pileup_DOWN", "Merge_Hist_Signal_PP8_Comb.root")

nominal_sys_mass, _ = fit_histogram(nominal_sys_path)
pileup_up_mass, _ = fit_histogram(pileup_up_path)
pileup_down_mass, _ = fit_histogram(pileup_down_path)

pileup_unc, pileup_method = calc_up_down_systematic(pileup_up_mass, pileup_down_mass, nominal_sys_mass)

print(f"\nPileup")
print(f"  nominal    = {nominal_sys_mass:.6f} GeV")
print(f"  pileup up  = {pileup_up_mass:.6f} GeV")
print(f"  pileup down= {pileup_down_mass:.6f} GeV")
print(f"  uncertainty = {pileup_unc:.6f} GeV  ({pileup_method})")

with open(f"{HOME}/lineshape_pileup_ForDolapo.txt", "w") as out:
    out.write(f"LineShape: {lineshape_unc:.6f} GeV (one-sided vs nominal AF)\n")
    out.write(f"Pileup: {pileup_unc:.6f} GeV ({pileup_method})\n")

print(f"\nSaved to {HOME}/lineshape_pileup_ForDolapo.txt")
