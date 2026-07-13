import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME = "/home/drojugbo"

# Read residual summaries for all 5 mass points
mass_list     = [171.0, 172.0, 172.5, 173.0, 174.0]
mass_str_list = ["171p0", "172p0", "172p5", "173p0", "174p0"]

residual_means  = []
residual_stds   = []
residual_errors = []

for mass_str in mass_str_list:
    fname = f"{HOME}/output_closure/residual_summary_{mass_str}_100k.txt"
    with open(fname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            residual_means.append(float(parts[1]))
            residual_errors.append(float(parts[2]))
            residual_stds.append(float(parts[3]))

residual_means  = np.array(residual_means)
residual_errors = np.array(residual_errors)
residual_stds   = np.array(residual_stds)

# Print summary table
print("Closure summary (100k experiments):")
print(f"{'m_true':>8} | {'mean residual':>15} | {'error on mean':>14} | {'std dev':>10}")
print("-" * 58)
for m, r, e, s in zip(mass_list, residual_means, residual_errors, residual_stds):
    print(f"{m:>8.1f} | {r:>+15.6f} | {e:>14.6f} | {s:>10.6f}")

# Largest deviation from zero — taken as method systematic uncertainty
largest_deviation = np.max(np.abs(residual_means))
print(f"\nLargest deviation: {largest_deviation:.6f} GeV ({largest_deviation*1000:.1f} MeV)")
print(f"This is taken as the systematic uncertainty on the method.")

# Closure plot using standard deviation as error bars
plt.figure(figsize=(8, 5))
plt.errorbar(mass_list, residual_means, yerr=residual_stds,
             fmt='o', color='blue', elinewidth=1.5, capsize=4, ms=6,
             label='Mean residual $\\pm$ std dev')
plt.axhline(0, color='black', ls='--', lw=1.5, label='Zero (ideal)')

# Add method uncertainty as text in upper left corner
plt.text(0.03, 0.95,
         f"Method uncertainty: {largest_deviation*1000:.1f} MeV",
         transform=plt.gca().transAxes,
         fontsize=10, color='red', va='top')

plt.xlim(170.5, 174.5)
plt.xlabel(r"$m_{top}^{true}$ [GeV]")
plt.ylabel(r"$\langle m_{top}^{fit} - m_{top}^{true} \rangle$ [GeV]")
plt.title("Closure plot (100k pseudo-experiments per mass point)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{HOME}/output_closure/closure_plot_100k.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved closure_plot_100k.png")
