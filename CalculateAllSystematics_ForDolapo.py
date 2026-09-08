import csv
import re
import numpy as np

HOME = "/home/drojugbo"
RESULTS_FILE = f"{HOME}/systematic_fit_results_ForDolapo.csv"
OUTPUT_FILE = f"{HOME}/systematic_uncertainty_table_ForDolapo.txt"

ONE_SIDED_SOURCES = [
    "Merged_MET_SoftTrk_ResoPara",
    "Merged_MET_SoftTrk_ResoPerp",
    "Merged_weight_nnlo_3d",
    "Merged_weight_MMHT",
]

SUFFIX_PATTERNS = [
    (r"__1up$", "up"),
    (r"__1down$", "down"),
    (r"_UP$", "up"),
    (r"_DOWN$", "down"),
    (r"_up(\[\d+\])?$", "up"),
    (r"_down(\[\d+\])?$", "down"),
    (r"05$", "down"),
    (r"20$", "up"),
]


def classify(name):
    # NNPDF alpha_s variation - checked first since it has its own suffix
    if name.endswith("_as_dw"):
        return name[:-len("_as_dw")] + "_as", "down"
    if name.endswith("_as_up"):
        return name[:-len("_as_up")] + "_as", "up"

    for pattern, direction in SUFFIX_PATTERNS:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            base = name[:match.start()]
            bracket = re.search(r"(\[\d+\])$", match.group())
            if bracket:
                base = base + bracket.group()
            return base, direction
    return None, None


results = {}
with open(RESULTS_FILE) as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = row["label"]
        results[label] = {
            "mass": float(row["fitted_mass_GeV"]),
            "success": row["fit_success"] == "True",
        }

if "nominal" not in results:
    raise RuntimeError("No nominal fit found in results file")

nominal_mass = results["nominal"]["mass"]
print(f"Nominal fitted mass: {nominal_mass:.6f} GeV\n")


def calc_up_down_systematic(up, down, nominal):
    lower, upper = min(up, down), max(up, down)
    if lower <= nominal <= upper:
        return abs(up - down) / 2.0, "half difference"
    delta_1 = up - nominal
    delta_2 = down - nominal
    if abs(delta_1) > abs(delta_2):
        return abs(delta_1), "largest shift (up)"
    return abs(delta_2), "largest shift (down)"


sources = {}
nnpdf_masses = []
one_sided_results = {}
unmatched = []

for label, info in results.items():
    if label == "nominal":
        continue

    if label.startswith("Merged_weight_NNPDF_Var_"):
        nnpdf_masses.append(info["mass"])
        continue

    if label in ONE_SIDED_SOURCES:
        one_sided_results[label] = info["mass"]
        continue

    base, direction = classify(label)
    if base is None:
        unmatched.append(label)
        continue

    sources.setdefault(base, {})
    sources[base][direction] = info["mass"]


rows = []

for base in sorted(sources.keys()):
    pair = sources[base]
    if "up" not in pair or "down" not in pair:
        rows.append((base, None, "incomplete pair - missing up or down"))
        continue

    systematic, method = calc_up_down_systematic(
        pair["up"], pair["down"], nominal_mass
    )
    rows.append((base, systematic, method))

for label, mass in one_sided_results.items():
    systematic = abs(mass - nominal_mass)
    if label == "Merged_weight_MMHT":
        method = "PDF set comparison, |fit - nominal|"
    else:
        method = "one-sided, |fit - nominal|"
    rows.append((label, systematic, method))

if nnpdf_masses:
    nnpdf_masses = np.array(nnpdf_masses)
    nnpdf_mean = nnpdf_masses.mean()
    nnpdf_rms = np.sqrt(np.mean((nnpdf_masses - nnpdf_mean) ** 2))
    rows.append(("NNPDF (RMS of variations)", nnpdf_rms,
                 f"RMS around mean of {len(nnpdf_masses)} variations"))

with open(OUTPUT_FILE, "w") as out:
    out.write(f"Nominal fitted mass: {nominal_mass:.6f} GeV\n")
    out.write("=" * 80 + "\n\n")
    out.write(f"{'Source':<55} {'Systematic (GeV)':<18} {'Method'}\n")
    out.write("-" * 80 + "\n")

    for name, value, method in rows:
        if value is None:
            out.write(f"{name:<55} {'MISSING':<18} {method}\n")
        else:
            out.write(f"{name:<55} {value:<18.6f} {method}\n")

    if unmatched:
        out.write("\nUnmatched labels (need review):\n")
        for name in unmatched:
            out.write(f"  {name}\n")

print(f"Computed systematics for {len([r for r in rows if r[1] is not None])} sources")
print(f"Incomplete pairs: {len([r for r in rows if r[1] is None])}")
print(f"Unmatched: {len(unmatched)}")
print(f"Saved table to {OUTPUT_FILE}")
