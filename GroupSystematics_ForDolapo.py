import re

HOME = "/home/drojugbo"
INPUT_FILE = f"{HOME}/systematic_uncertainty_table_ForDolapo.txt"
OUTPUT_FILE = f"{HOME}/systematic_uncertainty_grouped_ForDolapo.txt"

GROUP_RULES = [
    ("JES",     lambda name: name.startswith("Merged_JET_")),
    ("Lepton",  lambda name: (
        name.startswith("Merged_EG_")
        or name.startswith("Merged_MUON_")
        or name.startswith("Merged_weight_leptonSF")
    )),
    ("MET",     lambda name: name.startswith("Merged_MET_")),
    ("bTagSF",  lambda name: name.startswith("Merged_weight_bTagSF")),
]


def assign_group(name):
    for group_name, rule in GROUP_RULES:
        if rule(name):
            return group_name
    return None


rows = []
with open(INPUT_FILE) as f:
    for line in f:
        line = line.rstrip("\n")
        match = re.match(r"^(Merged_\S+|NNPDF\s*\(.*?\))\s+([\d.]+)\s+(.*)$", line)
        if match:
            name = match.group(1)
            value = float(match.group(2))
            method = match.group(3)
            rows.append((name, value, method))

print(f"Read {len(rows)} systematic sources from {INPUT_FILE}")

groups = {"JES": [], "Lepton": [], "MET": [], "bTagSF": []}
ungrouped = []

for name, value, method in rows:
    group_name = assign_group(name)
    if group_name:
        groups[group_name].append((name, value))
    else:
        ungrouped.append((name, value, method))

with open(OUTPUT_FILE, "w") as out:
    out.write("Grouped systematic uncertainties (ForDolapo)\n")
    out.write("=" * 70 + "\n\n")

    for group_name in ["JES", "Lepton", "MET", "bTagSF"]:
        members = groups[group_name]
        out.write(f"{group_name} group ({len(members)} sources)\n")
        out.write("-" * 70 + "\n")

        for name, value in members:
            out.write(f"    {name:<55} {value:.6f} GeV\n")

        quad_sum = sum(v ** 2 for _, v in members) ** 0.5 if members else 0.0
        out.write(f"  {group_name} combined (quadrature sum): {quad_sum:.6f} GeV\n\n")
        print(f"{group_name}: {len(members)} sources, combined = {quad_sum:.6f} GeV")

    out.write("\nUngrouped sources (kept individual)\n")
    out.write("-" * 70 + "\n")
    for name, value, method in ungrouped:
        out.write(f"    {name:<55} {value:.6f} GeV   {method}\n")

print(f"\nSaved grouped table to {OUTPUT_FILE}")
