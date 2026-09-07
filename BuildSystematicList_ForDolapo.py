import os
import re

SYS_FOLDER = ("/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_ForDolapo/"
              "Output_lepjets_Win_NewDnn_mlb_50_150_mw_50_110/Out_SYSLJ_FS")

OUTPUT_FILE = "/home/drojugbo/systematic_sources_list_ForDolapo.txt"

# folders to always exclude, matching Andrea's instructions
EXCLUDE_PREFIXES = [
    "Unmerged_",
    "Merged_JET_JER_",
    "Merged_weight_jvt",
    "Merged_weight_pileup",
    "Merged_nominal",
]
EXCLUDE_EXACT_NAMES = ["Merged_weight_leptonSF"]
EXCLUDE_EXACT = ["SubmissionFolder"]

# confirmed one-sided uncertainties from the old production - re-check
# these still apply here, since the folder set may differ slightly
ONE_SIDED_SOURCES = [
    "Merged_MET_SoftTrk_ResoPara",
    "Merged_MET_SoftTrk_ResoPerp",
    "Merged_weight_nnlo_3d",
]


def is_excluded(name):
    if name in EXCLUDE_EXACT or name in EXCLUDE_EXACT_NAMES:
        return True
    for prefix in EXCLUDE_PREFIXES:
        if name.startswith(prefix):
            return True
    return False


def classify(name):
    match = re.search(r"__1(up|down)$", name, re.IGNORECASE)
    if match:
        return name[:match.start()], match.group(1).lower()

    match = re.search(r"_(UP|DOWN)$", name)
    if match:
        return name[:match.start()], match.group(1).lower()

    match = re.search(r"_(up|down)(\[\d+\])?$", name, re.IGNORECASE)
    if match:
        direction = match.group(1).lower()
        index = match.group(2) or ""
        base = name[:match.start()] + index
        return base, direction

    if name.endswith("05"):
        return name[:-2], "down"
    if name.endswith("20"):
        return name[:-2], "up"

    # NNPDF alpha_s variation
    if name.endswith("_as_dw"):
        return name[:-len("_as_dw")] + "_as", "down"
    if name.endswith("_as_up"):
        return name[:-len("_as_up")] + "_as", "up"

    return None, None


folder_names = sorted(os.listdir(SYS_FOLDER))

sources = {}
nnpdf_variations = []
mmht_sources = []
one_sided_found = []
unclassified = []

for name in folder_names:
    full_path = os.path.join(SYS_FOLDER, name)
    if not os.path.isdir(full_path):
        continue
    if is_excluded(name):
        continue
    if not name.startswith("Merged_"):
        continue

    if name.startswith("Merged_weight_NNPDF_Var_"):
        nnpdf_variations.append(name)
        continue

    if name.startswith("Merged_weight_MMHT"):
        mmht_sources.append(name)
        continue

    if name in ONE_SIDED_SOURCES:
        one_sided_found.append(name)
        continue

    base, direction = classify(name)
    if base is None:
        unclassified.append(name)
        continue

    sources.setdefault(base, {})
    sources[base][direction] = name

with open(OUTPUT_FILE, "w") as out:
    out.write("Systematic sources with up/down components (ForDolapo)\n")
    out.write("=" * 70 + "\n\n")

    complete_count = 0
    incomplete_count = 0

    for base in sorted(sources.keys()):
        pair = sources[base]
        up_folder = pair.get("up", "MISSING")
        down_folder = pair.get("down", "MISSING")

        if "MISSING" in (up_folder, down_folder):
            flag = "  <-- incomplete pair"
            incomplete_count += 1
        else:
            flag = ""
            complete_count += 1

        out.write(f"{base}{flag}\n")
        out.write(f"    up:   {up_folder}\n")
        out.write(f"    down: {down_folder}\n\n")

    out.write("\nSummary\n")
    out.write("=" * 70 + "\n")
    out.write(f"Complete up/down pairs: {complete_count}\n")
    out.write(f"Incomplete pairs: {incomplete_count}\n")
    out.write(f"NNPDF variations: {len(nnpdf_variations)}\n")
    out.write(f"MMHT sources: {len(mmht_sources)}\n")
    out.write(f"One-sided uncertainties: {len(one_sided_found)}\n")
    out.write(f"Folders needing manual review: {len(unclassified)}\n")

    if one_sided_found:
        out.write("\nOne-sided uncertainties\n")
        out.write("=" * 70 + "\n")
        for name in sorted(one_sided_found):
            out.write(f"{name}\n")

    if mmht_sources:
        out.write("\nMMHT sources (new - not seen in old production)\n")
        out.write("=" * 70 + "\n")
        for name in sorted(mmht_sources):
            out.write(f"{name}\n")

    if nnpdf_variations:
        out.write(f"\nNNPDF variations ({len(nnpdf_variations)})\n")
        out.write("=" * 70 + "\n")
        for name in sorted(nnpdf_variations):
            out.write(f"{name}\n")

    if unclassified:
        out.write("\nFolders that need manual review\n")
        out.write("=" * 70 + "\n")
        for name in sorted(unclassified):
            out.write(f"{name}\n")

print(f"Found {len(sources)} systematic sources with up/down pairs")
print(f"Found {len(nnpdf_variations)} NNPDF variations")
print(f"Found {len(mmht_sources)} MMHT sources")
print(f"Found {len(one_sided_found)} one-sided uncertainties")
print(f"Found {len(unclassified)} folders needing manual review")
print(f"Saved to {OUTPUT_FILE}")
