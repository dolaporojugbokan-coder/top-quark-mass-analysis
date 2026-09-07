import os
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

OUTPUT_FILE = f"{HOME}/systematic_fit_results_ForDolapo.csv"
HIST_NAME = "h_mtop_param"

LOWER_BOUND = 168.0
UPPER_BOUND = 177.0

# folders to exclude, matching Andrea's instructions - same as
# BuildSystematicList_ForDolapo.py
EXCLUDE_PREFIXES = [
    "Unmerged_",
    "Merged_JET_JER_",
    "Merged_weight_jvt",
    "Merged_weight_pileup",
    "Merged_nominal",   # fitted separately as NOMINAL_FILE
]
EXCLUDE_EXACT_NAMES = ["Merged_weight_leptonSF"]
EXCLUDE_EXACT = ["SubmissionFolder"]


def is_excluded(name):
    if name in EXCLUDE_EXACT or name in EXCLUDE_EXACT_NAMES:
        return True
    for prefix in EXCLUDE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


# load the new ForDolapo parametrisation
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


# build the list of folders to fit
all_folder_names = sorted(os.listdir(SYS_FOLDER))

folders_to_fit = []
for name in all_folder_names:
    full_path = os.path.join(SYS_FOLDER, name)
    if not os.path.isdir(full_path):
        continue
    if is_excluded(name):
        continue
    if not name.startswith("Merged_"):
        continue
    folders_to_fit.append(name)

print(f"Found {len(folders_to_fit)} systematic folders to fit")

# overwrite the CSV at the start of a clean run
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "label", "root_file", "fitted_mass_GeV",
        "minimum_nll", "fit_success", "histogram_integral"
    ])

# fit the nominal first
print("\nFitting nominal...")
try:
    fit_result, integral = fit_histogram(NOMINAL_FILE)
    print(f"nominal: mass={fit_result.x:.6f} GeV, N={integral:.0f}")
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "nominal", NOMINAL_FILE, fit_result.x,
            fit_result.fun, fit_result.success, integral
        ])
except RuntimeError as error:
    print(f"nominal: FAILED - {error}")

# fit every systematic folder
failed = []
for i, folder_name in enumerate(folders_to_fit, 1):
    root_path = os.path.join(SYS_FOLDER, folder_name, SIGNAL_FILENAME)

    if not os.path.isfile(root_path):
        print(f"[{i}/{len(folders_to_fit)}] {folder_name}: "
              f"FILE NOT FOUND - {root_path}")
        failed.append(folder_name)
        continue

    try:
        fit_result, integral = fit_histogram(root_path)
        print(f"[{i}/{len(folders_to_fit)}] {folder_name}: "
              f"mass={fit_result.x:.6f} GeV, N={integral:.0f}")

        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                folder_name, root_path, fit_result.x,
                fit_result.fun, fit_result.success, integral
            ])

    except RuntimeError as error:
        print(f"[{i}/{len(folders_to_fit)}] {folder_name}: FAILED - {error}")
        failed.append(folder_name)

print(f"\n{len(folders_to_fit) - len(failed)}/{len(folders_to_fit)} "
      f"systematic fits completed successfully")
if failed:
    print(f"Failed folders: {failed}")

print(f"\nAll results saved to {OUTPUT_FILE}")
