import os
import csv
import ROOT
import numpy as np
import scipy.optimize as opt

REBIN_FACTOR = 1
HOME = "/home/drojugbo"

PARAM_FILE = f"{HOME}/output_parametrisation_ForDolapo/Param_mtop.txt"

HIST_NAME = "h_mtop_param"
LOWER_BOUND = 168.0
UPPER_BOUND = 177.0

OUTPUT_FILE = f"{HOME}/mc_generator_fit_results_ForDolapo.csv"

BASE_FS = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
           "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_NOM_FS/Merged_nominal")
BASE_AF = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
           "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_NOM_AF/Merged_nominal")

# hdamp is not yet available as a combined Signal_hdamp_Comb.root file
# in this production - only the ttbar-only version exists. Flagged and
# skipped rather than guessed at.
MC_SAMPLES = {
    "sample1_PP8_FS":         f"{BASE_FS}/Merge_Hist_Signal_PP8_Comb.root",
    "sample2_DS_FS":          f"{BASE_FS}/Merge_Hist_Signal_PP8_DS_Comb.root",
    "sample3_PP8_AF":         f"{BASE_AF}/Merge_Hist_Signal_PP8_Comb.root",
    "sample4_pthard1_AF":     f"{BASE_AF}/Merge_Hist_Signal_pthard1_Comb.root",
    "sample5_RecoilToTop_AF": f"{BASE_AF}/Merge_Hist_Signal_RecoilToTop_Comb.root",
    "sample6_PH713_AF":       f"{BASE_AF}/Merge_Hist_Signal_PH713_Comb.root",
    "sample7_CR0_AF":         f"{BASE_AF}/Merge_Hist_Signal_CR0_Comb.root",
    "sample8_CR1_AF":         f"{BASE_AF}/Merge_Hist_Signal_CR1_Comb.root",
    "sample9_CR2_AF":         f"{BASE_AF}/Merge_Hist_Signal_CR2_Comb.root",
    "sample10_FSRup_AF":      f"{BASE_AF}/Merge_Hist_Signal_FSRup_Comb.root",
    "sample11_FSRdown_AF":    f"{BASE_AF}/Merge_Hist_Signal_FSRdown_Comb.root",
    "sample12_UEup_AF":       f"{BASE_AF}/Merge_Hist_Signal_UE_up_Comb.root",
    "sample13_UEdown_AF":     f"{BASE_AF}/Merge_Hist_Signal_UE_dw_Comb.root",
    # sample14 (hdamp) intentionally omitted - Signal_hdamp_Comb.root
    # does not exist yet in this production, only Merge_Hist_ttbar_hdamp_Comb.root
}

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
print(f"Loaded {nbins} bins from {PARAM_FILE}\n")


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

    if REBIN_FACTOR > 1:
        histogram.Rebin(REBIN_FACTOR)

    if histogram.GetNbinsX() != nbins:
        raise RuntimeError(
            f"Histogram has {histogram.GetNbinsX()} bins, "
            f"but parametrisation has {nbins} bins"
        )

    data = np.array(
        [histogram.GetBinContent(i) for i in range(1, nbins + 1)]
    )

    if np.any(data < 0):
        negative_bins = np.where(data < 0)[0] + 1
        raise RuntimeError(
            f"Histogram contains negative bins {negative_bins.tolist()}: "
            f"{root_path}"
        )

    if data.sum() <= 0:
        raise RuntimeError(f"Histogram integral is zero or negative: {root_path}")

    fit_result = opt.minimize_scalar(
        lambda mass: negLogLik(mass, wdf, data),
        bounds=(LOWER_BOUND, UPPER_BOUND),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 1000},
    )

    if not fit_result.success:
        raise RuntimeError(
            f"Mass fit did not converge for {root_path}: {fit_result.message}"
        )

    return fit_result, data.sum()


results = {}
failed = {}

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "label", "root_file", "fitted_mass_GeV",
        "minimum_nll", "fit_success", "histogram_integral"
    ])

for label, root_path in MC_SAMPLES.items():
    if not os.path.isfile(root_path):
        print(f"{label}: FILE NOT FOUND - {root_path}")
        failed[label] = "file not found"
        continue
    try:
        fit_result, integral = fit_histogram(root_path)
        results[label] = fit_result.x
        print(f"{label}: mass={fit_result.x:.6f} GeV, N={integral:.0f}")

        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                label, root_path, fit_result.x,
                fit_result.fun, fit_result.success, integral
            ])

    except RuntimeError as error:
        print(f"{label}: FAILED - {error}")
        failed[label] = str(error)

print(f"\nSaved fit results to {OUTPUT_FILE}")

print("\nMC generator systematic uncertainties (ForDolapo)")
print("=" * 60)


def diff(a, b):
    return abs(results[a] - results[b])


def up_down_nominal(up_label, down_label, nominal_label):
    up = results[up_label]
    down = results[down_label]
    nominal = results[nominal_label]
    lower, upper = min(up, down), max(up, down)
    if lower <= nominal <= upper:
        return abs(up - down) / 2.0
    delta_1 = up - nominal
    delta_2 = down - nominal
    return abs(delta_1) if abs(delta_1) > abs(delta_2) else abs(delta_2)


def show_and_diff(name, label_a, label_b):
    val = diff(label_a, label_b)
    print(f"{name}")
    print(f"  {label_a} = {results[label_a]:.6f} GeV")
    print(f"  {label_b} = {results[label_b]:.6f} GeV")
    print(f"  uncertainty = {val:.6f} GeV\n")
    return val


summary_rows = []

try:
    val = show_and_diff("tW interference (DR vs DS)", "sample1_PP8_FS", "sample2_DS_FS")
    summary_rows.append(("tW interference (DR vs DS)", val))
except KeyError:
    summary_rows.append(("tW interference (DR vs DS)", None))

try:
    val = show_and_diff("Matrix-element matching", "sample3_PP8_AF", "sample4_pthard1_AF")
    summary_rows.append(("Matrix-element matching", val))
except KeyError:
    summary_rows.append(("Matrix-element matching", None))

try:
    val = show_and_diff("Recoil", "sample3_PP8_AF", "sample5_RecoilToTop_AF")
    summary_rows.append(("Recoil", val))
except KeyError:
    summary_rows.append(("Recoil", None))

try:
    val = show_and_diff("Parton shower and hadronisation", "sample3_PP8_AF", "sample6_PH713_AF")
    summary_rows.append(("Parton shower and hadronisation", val))
except KeyError:
    summary_rows.append(("Parton shower and hadronisation", None))

try:
    val_a = diff("sample7_CR0_AF", "sample8_CR1_AF")
    val_b = diff("sample7_CR0_AF", "sample9_CR2_AF")
    chosen = max(val_a, val_b)
    print("Colour reconnection")
    print(f"  CR0 = {results['sample7_CR0_AF']:.6f} GeV")
    print(f"  CR1 = {results['sample8_CR1_AF']:.6f} GeV")
    print(f"  CR2 = {results['sample9_CR2_AF']:.6f} GeV")
    print(f"  |CR1 - CR0| = {val_a:.6f} GeV")
    print(f"  |CR2 - CR0| = {val_b:.6f} GeV")
    print(f"  uncertainty = larger = {chosen:.6f} GeV\n")
    summary_rows.append(("Colour reconnection", chosen))
except KeyError:
    summary_rows.append(("Colour reconnection", None))

try:
    up = results["sample10_FSRup_AF"]
    down = results["sample11_FSRdown_AF"]
    nominal = results["sample3_PP8_AF"]
    val = up_down_nominal("sample10_FSRup_AF", "sample11_FSRdown_AF", "sample3_PP8_AF")
    print("FSR")
    print(f"  nominal AF = {nominal:.6f} GeV")
    print(f"  FSR up     = {up:.6f} GeV")
    print(f"  FSR down   = {down:.6f} GeV")
    print(f"  uncertainty = {val:.6f} GeV\n")
    summary_rows.append(("FSR", val))
except KeyError:
    summary_rows.append(("FSR", None))

try:
    up = results["sample12_UEup_AF"]
    down = results["sample13_UEdown_AF"]
    nominal = results["sample3_PP8_AF"]
    val = up_down_nominal("sample12_UEup_AF", "sample13_UEdown_AF", "sample3_PP8_AF")
    print("UE (underlying event)")
    print(f"  nominal AF = {nominal:.6f} GeV")
    print(f"  UE up      = {up:.6f} GeV")
    print(f"  UE down    = {down:.6f} GeV")
    print(f"  uncertainty = {val:.6f} GeV\n")
    summary_rows.append(("UE (underlying event)", val))
except KeyError:
    summary_rows.append(("UE (underlying event)", None))

# hdamp intentionally not attempted - Signal_hdamp_Comb.root missing
summary_rows.append(("hdamp", None))

print("Final summary")
print("=" * 60)
for name, value in summary_rows:
    if value is None:
        note = " (BLOCKED - Signal_hdamp_Comb.root not yet produced)" if name == "hdamp" else " (MISSING - check failed samples above)"
        print(f"{name:<40} PENDING{note}")
    else:
        print(f"{name:<40} {value:.6f} GeV")

SUMMARY_FILE = f"{HOME}/mc_generator_uncertainty_table_ForDolapo.txt"
with open(SUMMARY_FILE, "w") as out:
    out.write("MC generator uncertainties (ForDolapo)\n")
    out.write("=" * 60 + "\n\n")
    for name, value in summary_rows:
        if value is None:
            out.write(f"{name:<40} PENDING\n")
        else:
            out.write(f"{name:<40} {value:.6f} GeV\n")

print(f"\nSaved summary to {SUMMARY_FILE}")
if failed:
    print(f"\nFailed/missing samples: {failed}")
