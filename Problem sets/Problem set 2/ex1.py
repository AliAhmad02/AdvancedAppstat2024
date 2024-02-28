import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import sys

sys.path.append("../AdvancedAppstat")
from utilities import num_to_latex_str, num_err_to_latex_str

# Requires scienceplots style installation
import scienceplots

plt.style.use(["science"])


def neutrino_pdf_norm(E, M):
    T = 1.0 * 1.057e13 / M
    # The normalization from 0 to infinity
    norm = 1 / (T * np.log(2) / 3)
    f = norm * (np.exp(E / T) - 1) / ((np.exp(E / T) + 3) * (np.exp(E / T) + 1))
    return f


m1 = 2.5e11
m2 = 4e11
m3 = 9e11

neutrino_lin_min = 0
neutrino_lin_max = 250
neutrino_lin_N = 10_000

neutrino_lin = np.linspace(neutrino_lin_min, neutrino_lin_max, neutrino_lin_N)
pdf_m1 = neutrino_pdf_norm(neutrino_lin, m1)
pdf_m2 = neutrino_pdf_norm(neutrino_lin, m2)
pdf_m3 = neutrino_pdf_norm(neutrino_lin, m3)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(
    neutrino_lin,
    pdf_m1,
    label=rf"PDF for Black hole mass {num_to_latex_str(m1, 2)}g",
    color="aqua",
)
ax.plot(
    neutrino_lin,
    pdf_m2,
    label=rf"PDF for Black hole mass {num_to_latex_str(m2, 2)}g",
    color="dodgerblue",
)
ax.plot(
    neutrino_lin,
    pdf_m3,
    label=rf"PDF for Black hole mass {num_to_latex_str(m3, 2)}g",
    color="midnightblue",
)
ax.set_xlabel("Neutrino Energy [GeV]", fontsize=15)
ax.set_ylabel("Probability", fontsize=15)
plt.legend(frameon=False, fontsize=13)
plt.show()

sim_neutrino_path = "https://www.nbi.dk/~koskinen/Teaching/data/neutrino_energies.csv"
sim_neutrino_data = pd.read_csv(sim_neutrino_path).T.values[0]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    sim_neutrino_data,
    bins=50,
    density=True,
    color="tomato",
    histtype="step",
    label="Simulated data",
)
ax.plot(
    neutrino_lin,
    pdf_m1,
    label=rf"PDF for Black hole mass {num_to_latex_str(m1, 2)}g",
    color="aqua",
)
ax.plot(
    neutrino_lin,
    pdf_m2,
    label=rf"PDF for Black hole mass {num_to_latex_str(m2, 2)}g",
    color="dodgerblue",
)
ax.plot(
    neutrino_lin,
    pdf_m3,
    label=rf"PDF for Black hole mass {num_to_latex_str(m3, 2)}g",
    color="midnightblue",
)
ax.set_xlabel("Neutrino Energy [GeV]", fontsize=15)
ax.set_ylabel("Probability", fontsize=15)
plt.legend(frameon=False, fontsize=13)
plt.show()
print("\nEyeball guess for neutrino mass, around 6e+11g")

N_raster = 10_000
bh_mass_raster_lin = np.linspace(m2, m3, N_raster)
nll_data_obj = UnbinnedNLL(sim_neutrino_data, neutrino_pdf_norm)
# Multiply by -1/2 since the function gives us -2*LLH
llh_mass_raster_vals = (
    -1 / 2 * np.array([nll_data_obj(mass) for mass in bh_mass_raster_lin])
)
llh_mass_raster_max_idx = np.argmax(llh_mass_raster_vals)
llh_mass_raster_max_val = np.max(llh_mass_raster_vals)
mass_raster_max_llh = bh_mass_raster_lin[llh_mass_raster_max_idx]
mass_raster_max_llh_1sig = np.where(
    llh_mass_raster_vals >= (llh_mass_raster_max_val - 0.5)
)[0]
mass_raster_max_llh_err_left_idx = mass_raster_max_llh_1sig[0]
mass_raster_max_llh_err_right_idx = mass_raster_max_llh_1sig[-1]

mass_raster_max_llh_err_left = (
    mass_raster_max_llh - bh_mass_raster_lin[mass_raster_max_llh_err_left_idx]
)
mass_raster_max_llh_err_right = (
    bh_mass_raster_lin[mass_raster_max_llh_err_right_idx] - mass_raster_max_llh
)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(bh_mass_raster_lin, llh_mass_raster_vals, color="dodgerblue", lw=1.5)
ax.set_xlabel(r"$M_{BH}$ [g]", fontsize=17)
ax.set_ylabel(r"$\ln{{(\mathcal{L})}}$", fontsize=17)
ax.axvline(mass_raster_max_llh, lw=1.5, color="black", linestyle="dashed")
ax.text(
    0.45,
    0.2,
    rf"$\hat{{M}}_{{BH}}[\ln({{\mathcal{{L}}}}_{{max}})\sim{llh_mass_raster_max_val:.1f}]$={num_err_to_latex_str(mass_raster_max_llh, mass_raster_max_llh_err_left, 2)} g",
    transform=ax.transAxes,
    va="center",
    ha="center",
    fontsize=12,
    bbox=dict(fc="white", edgecolor="black"),
)
plt.show()

pdf_max_llh_mass = neutrino_pdf_norm(neutrino_lin, mass_raster_max_llh)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    sim_neutrino_data,
    bins=50,
    density=True,
    color="tomato",
    histtype="step",
    label="Simulated data",
)
ax.plot(
    neutrino_lin,
    pdf_max_llh_mass,
    label=f"PDF for MLE of black hole mass",
    color="dodgerblue",
)
ax.set_xlabel("Neutrino Energy [GeV]", fontsize=15)
ax.set_ylabel("Probability", fontsize=15)
ax.set_xlim(0, 200)
plt.legend(frameon=False, fontsize=13)
plt.show()

llh_m1 = -1 / 2 * nll_data_obj(m1)
llh_m2 = -1 / 2 * nll_data_obj(m2)
llh_m3 = -1 / 2 * nll_data_obj(m3)
print(
    f"\nLikelihood for proposed masses",
    f"\nm1: {llh_m1}\nm2: {llh_m2}\nm3: {llh_m3}",
)

minuit_obj_data = Minuit(nll_data_obj, M=mass_raster_max_llh)
minuit_obj_data.limits["M"] = (4e11, 7e11)
minuit_obj_data.migrad()
minuit_obj_data.hesse()
minuit_mass_estimate = minuit_obj_data.values[0]
minuit_mass_err = minuit_obj_data.errors[0]

print(f"\nBH mass minuit estimate {minuit_mass_estimate:.2e} +/- {minuit_mass_err:.2e}")
