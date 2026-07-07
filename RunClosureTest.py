import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME = "/home/drojugbo"

# Read the closure fit results saved by RunLikelihood_scipy.py
# Each line contains: m_input, m_measured, uncertainty
mass_input    = []
mass_measured = []
uncertainties = []

with open(f"{HOME}/output_likelihood/closure_fit_scipy.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.strip().split()
        mass_input.append(float(parts[0]))
        mass_measured.append(float(parts[1]))
        uncertainties.append(float(parts[2]))

# Bias = fitted mass - input mass to match the pull convention
# pull = (m_fit - m_true) / sigma
bias_values = [m_m - m_i for m_i, m_m in zip(mass_input, mass_measured)]

# Print individual closure residuals and expected pull per mass point
print("\nClosure summary:")
print(f"{'m_input':>8} {'m_fit':>8} {'residual':>10} {'sigma':>8} {'expected pull':>14}")
for m_i, m_m, sig in zip(mass_input, mass_measured, uncertainties):
    residual      = m_m - m_i
    expected_pull = residual / sig
    print(f"{m_i:>8.1f} {m_m:>8.4f} {residual:>+10.4f} {sig:>8.4f} {expected_pull:>+14.3f}")

# Plot bias vs input mass
# Individual points show whether the method closes at each mass
plt.figure(figsize=(7, 5))
plt.errorbar(mass_input, bias_values, yerr=uncertainties,
             fmt='o', color='b', elinewidth=1.5, capsize=4, ms=6)
plt.axhline(0, color='black', ls=':', lw=1)
plt.xlabel(r"$m_{top}^{input}$ [GeV]")
plt.ylabel(r"$m_{top}^{fit} - m_{top}^{input}$ [GeV]")
plt.tight_layout()
plt.savefig(f"{HOME}/output_closure/bias_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved bias_plot.png")
