slopes = []
intercepts = []
with open("/home/drojugbo/Param_mtop.txt", "r") as f:
    for line in f:
        parts = line.strip().split(",")
        slopes.append(float(parts[1].strip()))
        intercepts.append(float(parts[2].strip()))
print(f"first slope: {slopes[0]}")
print(f"first intercept: {intercepts[0]}")

import ROOT

folder = "/data/aknue/Output_212247_MASS_SwitchToFS_MPP_NewDNN2_onlyCP/Output_lepjets_Win_mlb_mw2/Out_NOM_FS/Merged_nominal"
HOME = "/home/drojugbo"
f = ROOT.TFile.Open(f"{folder}/Merge_Hist_Signal_PP8_Comb.root")
hist = f.Get("h_mtop_param")
hist.SetDirectory(0)


nbins = hist.GetNbinsX()
xmin = hist.GetXaxis().GetXmin()
xmax = hist.GetXaxis().GetXmax()

print(f"nbins = {nbins}")
print(f"xmin = {xmin}")
print(f"xmax = {xmax}")

def create_histogram(hist_template):
    nbins = hist_template.GetNbinsX()
    xmin = hist_template.GetXaxis().GetXmin()
    xmax = hist_template.GetXaxis().GetXmax()
    new_hist = ROOT.TH1D("new_hist", "new_hist", nbins, xmin, xmax)
    return new_hist
new_hist = create_histogram(hist)
print("new histogram created!")

for i in range(nbins):
    value = slopes[i] * 171 + intercepts[i]
    new_hist.SetBinContent(i+1, value)
new_hist.Scale(1.0 / new_hist.Integral())
print("histogram filled!")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mass_list = [171.0, 172.0, 172.5, 173.0, 174.0]
file_list = [
    "Merge_Hist_Signal_PP8_171_Comb.root",
    "Merge_Hist_Signal_PP8_172_Comb.root",
    "Merge_Hist_Signal_PP8_Comb.root",
    "Merge_Hist_Signal_PP8_173_Comb.root",
    "Merge_Hist_Signal_PP8_174_Comb.root"
]
x = [new_hist.GetBinCenter(i) for i in range(1, nbins+1)]
for mass, filename in zip(mass_list, file_list):
    f_mc   = ROOT.TFile.Open(f"{folder}/{filename}")
    h_mc   = f_mc.Get("h_mtop_param")
    h_mc.SetDirectory(0)
    h_mc.Scale(1.0 / h_mc.Integral())
    f_mc.Close()

    y_mc  = [h_mc.GetBinContent(i) for i in range(1, nbins+1)]
    err_mc = [h_mc.GetBinError(i)  for i in range(1, nbins+1)]

    # build parametrized template at this mass
    y_new = [slopes[i] * mass + intercepts[i] for i in range(nbins)]
    total = sum(y_new)
    y_new = [v / total for v in y_new]

    ratio_f   = [y_new[i] / y_mc[i]   if y_mc[i]  != 0 else 0 for i in range(nbins)]
    ratio_err = [err_mc[i] / y_mc[i]  if y_mc[i]  != 0 else 0 for i in range(nbins)]

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0}, sharex=True)

    ax1.plot(x, y_mc,  'b',   label=f'{mass} GeV (MC)', drawstyle='steps-mid')
    ax1.fill_between(x,
                     [y_mc[i] - err_mc[i] for i in range(nbins)],
                     [y_mc[i] + err_mc[i] for i in range(nbins)],
                     step='mid', facecolor='none', edgecolor='b',
                     hatch='///', linewidth=0.5, label='stat. unc.')
    ax1.plot(x, y_new, 'r--', label=f'f({mass})', linewidth=2, drawstyle='steps-mid')
    ax1.set_ylabel("Normalised events")
    ax1.legend()

    ax2.plot(x, ratio_f, 'r', drawstyle='steps-mid')
    ax2.errorbar(x, ratio_f, yerr=ratio_err, fmt='none',
                 ecolor='r', elinewidth=0.5, capsize=0)
    ax2.axhline(y=1, color='black', linewidth=0.8, ls='--')
    ax2.set_ylabel(f"f({mass}) / MC")
    ax2.set_xlabel("$m_{{lb}}$ [GeV]")

    mass_str = str(mass).replace('.', 'p')
    plt.savefig(f"{HOME}/closure_{mass_str}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved closure_{mass_str}.png")
