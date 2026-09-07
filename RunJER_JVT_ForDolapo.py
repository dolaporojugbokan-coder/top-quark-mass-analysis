import os
import re
import csv
import ROOT
import numpy as np
import scipy.optimize as opt

HOME = "/home/drojugbo"

PARAM_FILE = f"{HOME}/output_parametrisation_ForDolapo/Param_mtop.txt"

SYS_FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
              "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_SYSLJ_FS")

NOMINAL_FILE = os.path.join(SYS_FOLDER, "Merged_nominal", "Merge_Hist_Signal_PP8_Comb.root")
SIGNAL_FILENAME = "Merge_Hist_Signal_PP8_Comb.root"

OUTPUT_FIT_FILE = f"{HOME}/jer_jvt_fit_results_ForDolapo.csv"
OUTPUT_SUMMARY_FILE = f"{HOME}/jer_jvt_uncertainty_ForDolapo.txt"

HIST_NAME = "h_mtop_param"
LOWER_BOUND = 168.0
UPPER_BOUND = 177.0


# load the ForDolapo parametrisation
slopes = []
intercepts = []
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
print(f"Loaded {nbins} bins from {PARAM_FILE}")


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
    term[nonzero] += (
        observed[nonzero] * np.log(observed[nonzero] / expected[nonzero])
    )
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
        raise RuntimeError(
            f"Histogram has {histogram.GetNbinsX()} bins, "
            f"but parametrisation has {nbins} bins: {root_path}"
        )

    data = np.array(
        [histogram.GetBinContent(i) for i in range(1, nbins + 1)]
    )

    if np.any(data < 0):
        raise RuntimeError(f"Negative bin content: {root_path}")
    if data.sum() <= 0:
        raise RuntimeError(f"Non-positive integral: {root_path}")

    fit_result = opt.minimize_scalar(
        lambda mass: negLogLik(mass, wdf, data),
        bounds=(LOWER_BOUND, UPPER_BOUND),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )

    if not fit_result.success:
        raise RuntimeError(f"Fit did not converge for {root_path}: {fit_result.message}")

    return fit_result, data.sum()


# find every JER folder, and the two JVT folders
all_names = sorted(os.listdir(SYS_FOLDER))

jer_folders = [
    n for n in all_names
    if n.startswith("Merged_JET_JER_")
    and "MixedMCPD" in n
    and os.path.isdir(os.path.join(SYS_FOLDER, n))
]
jvt_folders = [
    n for n in all_names
    if n.startswith("Merged_weight_jvt_") and os.path.isdir(os.path.join(SYS_FOLDER, n))
]

print(f"Found {len(jer_folders)} JER folders")
print(f"Found {len(jvt_folders)} JVT folders")


def classify(name):
    # find __1up or __1down, keep base name plus any trailing suffix
    # (e.g. __MixedMCPD or _PseudoData) as part of the base, so different
    # flavours of the same NP number stay as separate sources
    match = re.search(r"__1(up|down)", name, re.IGNORECASE)
    if not match:
        return None, None
    direction = match.group(1).lower()
    base = name[:match.start()] + name[match.end():]
    return base, direction


# fit nominal once
print("\nFitting nominal...")
nominal_fit, nominal_integral = fit_histogram(NOMINAL_FILE)
nominal_mass = nominal_fit.x
print(f"nominal: mass={nominal_mass:.6f} GeV, N={nominal_integral:.0f}")

results = {}
failed = []

with open(OUTPUT_FIT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "root_file", "fitted_mass_GeV", "fit_success", "histogram_integral"])
    writer.writerow(["nominal", NOMINAL_FILE, nominal_mass, nominal_fit.success, nominal_integral])

all_folders_to_fit = [(n, "JER") for n in jer_folders] + [(n, "JVT") for n in jvt_folders]

for i, (folder_name, category) in enumerate(all_folders_to_fit, 1):
    root_path = os.path.join(SYS_FOLDER, folder_name, SIGNAL_FILENAME)

    if not os.path.isfile(root_path):
        print(f"[{i}/{len(all_folders_to_fit)}] {folder_name}: FILE NOT FOUND")
        failed.append(folder_name)
        continue

    try:
        fit_result, integral = fit_histogram(root_path)
        results[folder_name] = fit_result.x
        print(f"[{i}/{len(all_folders_to_fit)}] {folder_name}: mass={fit_result.x:.6f} GeV")

        with open(OUTPUT_FIT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([folder_name, root_path, fit_result.x, fit_result.success, integral])

    except RuntimeError as error:
        print(f"[{i}/{len(all_folders_to_fit)}] {folder_name}: FAILED - {error}")
        failed.append(folder_name)

print(f"\n{len(results)}/{len(all_folders_to_fit)} fits completed successfully")
if failed:
    print(f"Failed: {failed}")


def calc_up_down_systematic(up, down, nominal):
    lower, upper = min(up, down), max(up, down)
    if lower <= nominal <= upper:
        return abs(up - down) / 2.0
    delta_1 = up - nominal
    delta_2 = down - nominal
    return abs(delta_1) if abs(delta_1) > abs(delta_2) else abs(delta_2)


# group JER folders into up/down pairs by base name
jer_sources = {}
for name in jer_folders:
    if name not in results:
        continue
    base, direction = classify(name)
    if base is None:
        continue
    jer_sources.setdefault(base, {})
    jer_sources[base][direction] = results[name]

jer_values = []
jer_detail = []
for base in sorted(jer_sources.keys()):
    pair = jer_sources[base]
    if "up" not in pair or "down" not in pair:
        jer_detail.append((base, None))
        continue
    val = calc_up_down_systematic(pair["up"], pair["down"], nominal_mass)
    jer_values.append(val)
    jer_detail.append((base, val))

jer_combined = np.sqrt(np.sum(np.array(jer_values) ** 2)) if jer_values else 0.0

# JVT: single up/down pair
jvt_value = None
if "Merged_weight_jvt_UP" in results and "Merged_weight_jvt_DOWN" in results:
    jvt_value = calc_up_down_systematic(
        results["Merged_weight_jvt_UP"], results["Merged_weight_jvt_DOWN"], nominal_mass
    )

print(f"\nJER: {len(jer_values)} sources, combined (quadrature) = {jer_combined:.6f} GeV")
if jvt_value is not None:
    print(f"JVT: {jvt_value:.6f} GeV")
else:
    print("JVT: could not compute (missing up or down)")

with open(OUTPUT_SUMMARY_FILE, "w") as out:
    out.write("JER and JVT systematic uncertainties (ForDolapo)\n")
    out.write("=" * 70 + "\n\n")
    out.write(f"Nominal fitted mass: {nominal_mass:.6f} GeV\n\n")

    out.write(f"JER sub-components ({len(jer_detail)})\n")
    out.write("-" * 70 + "\n")
    for base, val in jer_detail:
        if val is None:
            out.write(f"    {base:<55} MISSING (incomplete pair)\n")
        else:
            out.write(f"    {base:<55} {val:.6f} GeV\n")
    out.write(f"\nJER combined (quadrature sum): {jer_combined:.6f} GeV\n\n")

    if jvt_value is not None:
        out.write(f"JVT: {jvt_value:.6f} GeV\n")
    else:
        out.write("JVT: could not compute\n")

print(f"\nSaved to {OUTPUT_SUMMARY_FILE}")
