import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec, dblquad
import scienceplots

plt.style.use("science")


def voigt_pdf(lam, lam_0, gamma, b):
    c = 3e5
    func = (
        lambda v, lam, lam_0, gamma, b: gamma
        / (np.pi ** (3 / 2) * b)
        * np.exp(-(v**2) / b**2)
        / ((lam - lam_0 - lam_0 * v / c) ** 2 + gamma**2)
    )
    f, _ = quad_vec(func, -np.inf, np.inf, args=(lam, lam_0, gamma, b))
    norm, _ = dblquad(func, 6300, 6400, -np.inf, np.inf, args=(lam_0, gamma, b))
    return f / norm


ga_iterations = (
    np.genfromtxt("AdvancedAppstat/Problem sets/Project/Data Files/GA_iterations.txt")
    / 2
)
MC_data = np.genfromtxt("AdvancedAppstat/Problem sets/Project/Data Files/MC_data.txt")
minuit_initial_llh = -46259.8940762004 / 2
minuit_initial_params = [5.00e3, 3.0, 24]
minuit_initial_params_err = [0.05e3, 2.5, 17]
true_params = [6350, 1, 17]
true_params_llh = -24123.015154305256 / 2
ga_fit_llh = (-24638.453811631844) / 2
ga_params = [6.34996375e03, 6.34554601e-01, 9.46659796e01]
ga_fit_llh_extra = -27633.09182490352 / 2
ga_iterations_extra = (
    np.genfromtxt(
        "AdvancedAppstat/Problem sets/Project/Data Files/GA_iterations_extra.txt"
    )
    / 2
)
ga_params_extra = [6.35045899e03, 3.43201575e-01, 2.56185987e02]
minuit_ga_llh = (-24122.281341168993) / 2
minuit_ga_params = [6.350004e3, 0.979, 18]
minuit_ga_params_err = [0.000021e3, 0.030, 6]
voigt_pdf_lin = np.linspace(6300, 6400, 10_000)
voigt_pdf_minuit_initial = voigt_pdf(voigt_pdf_lin, *minuit_initial_params)
voigt_pdf_ga = voigt_pdf(voigt_pdf_lin, *ga_params)
voigt_pdf_ga_minuit = voigt_pdf(voigt_pdf_lin, *minuit_ga_params)
iteration_nums = np.arange(1, 201)

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.hist(
    MC_data, bins=200, range=(6320, 6380), histtype="stepfilled", color="dodgerblue"
)
ax.set_xlabel("$\lambda$ [Å]", fontsize=17)
ax.set_ylabel("Flux [A.U.]", fontsize=17)
plt.savefig(
    "AdvancedAppstat/Problem sets/Project/Figures/MC_data.png",
    dpi=500,
    bbox_inches="tight",
)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.hist(
    MC_data,
    bins=200,
    range=(6320, 6380),
    density=True,
    histtype="stepfilled",
    color="dodgerblue",
    label="MC data (density)",
)
ax.plot(
    voigt_pdf_lin,
    voigt_pdf_minuit_initial,
    color="orange",
    label="iMinuit fit",
    lw=2,
)
ax.plot(voigt_pdf_lin, voigt_pdf_ga_minuit, color="black", label="GA+iMinuit fit", lw=2)
ax.plot(voigt_pdf_lin, voigt_pdf_ga, "--", color="red", label="GA fit", lw=2.5)
ax.set_xlabel("$\lambda$ [Å]", fontsize=17)
ax.set_ylabel("Probability density", fontsize=17)
ax.set_xlim(6320, 6380)
ax.legend(frameon=False, fontsize=15)
plt.savefig(
    "AdvancedAppstat/Problem sets/Project/Figures/Fit_comparisons.png",
    dpi=500,
    bbox_inches="tight",
)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(
    iteration_nums,
    true_params_llh - ga_iterations,
    color="dodgerblue",
    lw=1.5,
    label="First run",
)
# ax.plot(iteration_nums, true_params_llh - ga_iterations_extra, color="red", lw=1.5, label="Extra run")
ax.set_xlabel("Generation", fontsize=17)
ax.set_ylabel(
    "$\ln{{\mathcal{{L}}_{{true}}}}-\ln{{\mathcal{{L}}_{{max}}}}$", fontsize=17
)
plt.savefig(
    "AdvancedAppstat/Problem sets/Project/Figures/GA_evolution.png",
    dpi=500,
    bbox_inches="tight",
)
plt.show()
