import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME = "/home/drojugbo"

# Read closure_fit.txt
mass_input   = []
mass_measured = []
uncertainties = []

with open(f"{HOME}/output_likelihood/closure_fit.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        mass_input.append(float(parts[0]))
        mass_measured.append(float(parts[1]))
        uncertainties.append(float(parts[2]))

# Calculate bias
bias_values = [m_i - m_m for m_i, m_m in zip(mass_input, mass_measured)]

# Weighted mean
weights    = [1.0 / (e**2) for e in uncertainties]
mean_bias  = sum(w * b for w, b in zip(weights, bias_values)) / sum(weights)
mean_error = 1.0 / (sum(weights)**0.5)

print(f"Fitted bias: {mean_bias:+.3f} +/- {mean_error:.3f} GeV")

# Plot
plt.figure(figsize=(7, 5))
plt.errorbar(mass_input, bias_values, yerr=uncertainties,
             fmt='o', color='b', elinewidth=1.5, capsize=4, ms=6)
plt.axhline(mean_bias, color='r', ls='--', lw=1.5,
            label=f'fit: {mean_bias:+.3f} $\\pm$ {mean_error:.3f} GeV')
plt.axhline(0, color='black', ls=':', lw=1)
plt.xlabel(r"$m_{top}^{input}$ [GeV]")
plt.ylabel(r"$m_{top}^{input} - m_{top}^{fit}$ [GeV]")
plt.legend()
plt.tight_layout()
plt.savefig(f"{HOME}/output_closure/bias_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved bias_plot.png")
