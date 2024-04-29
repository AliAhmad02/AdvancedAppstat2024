import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
from nestle import sample
from nestle import mean_and_cov
from scipy.integrate import quad
import corner
from scipy.stats import ks_1samp, poisson, binom, norm, chi2
import scienceplots

plt.style.use("science")

data_path = "AdvancedAppstat/Exam_2024/Data_Files/Exam_2024_Prob1.txt"
data = pd.read_csv(
    data_path, delimiter="\s", engine="python", names=["A", "B", "C", "D", "E", "F"]
)
col1, _, col3, col4, _, _ = data.T.values

col1_min, col1_max = 6, 11


def sine_exp(x, a, b, c):
    return np.sin(a * x) + c * np.exp(b * x) + 1


def sine_exp_pdf(x, a, b, c):
    norm, _ = quad(sine_exp, col1_min, col1_max, args=(a, b, c))
    return sine_exp(x, a, b, c) / norm


def sine_exp_cdf(x, a, b, c):
    cdf = [quad(sine_exp_pdf, col1_min, upper, args=(a, b, c))[0] for upper in x]
    return np.array(cdf)


def poisson_pdf(k, lam):
    return poisson(lam).pmf(k)


def binom_pdf(k, n, p):
    return binom.pmf(k, n=n, p=p)


def prior_transform_sine_exp(theta):
    return np.array([theta[0] * 20 - 10, theta[1] * 20 - 10, theta[2] * 4000 + 4000])


def loglike_sine_exp(theta):
    return np.sum(np.log(sine_exp_pdf(col1, theta[0], theta[1], theta[2])))


def gauss_pdf(x, mu, sigma):
    return norm.pdf(x, loc=mu, scale=sigma)


res_sine_exp = sample(
    loglike_sine_exp, prior_transform_sine_exp, 3, method="single", npoints=300
)
p_sine_exp, cov_sine_exp = mean_and_cov(res_sine_exp.samples, res_sine_exp.weights)
a_sine_exp_fit, b_sine_exp_fit, c_sine_exp_fit = p_sine_exp
a_sine_exp_fit_err, b_sine_exp_fit_err, c_sine_exp_fit_err = np.sqrt(
    np.diag(cov_sine_exp)
)

corner.corner(
    res_sine_exp.samples,
    weights=res_sine_exp.weights,
    range=[
        (
            a_sine_exp_fit - 3 * a_sine_exp_fit_err,
            a_sine_exp_fit + 3 * a_sine_exp_fit_err,
        ),
        (
            b_sine_exp_fit - 3 * b_sine_exp_fit_err,
            b_sine_exp_fit + 3 * b_sine_exp_fit_err,
        ),
        (4000, 8000),
    ],
    show_titles=True,
    labels=["a", "b", "c"],
)
plt.show()

print(res_sine_exp.summary())
print(
    f"\na: {p_sine_exp[0]} +/- {np.sqrt(cov_sine_exp[0, 0])}\nb: {p_sine_exp[1]} +/- {np.sqrt(cov_sine_exp[1, 1])}\nc: {p_sine_exp[2]} +/- {np.sqrt(cov_sine_exp[2, 2]) }"
)
sine_exp_ks_obj = ks_1samp(
    col1, sine_exp_cdf, args=(a_sine_exp_fit, b_sine_exp_fit, c_sine_exp_fit)
)
sine_exp_pval = sine_exp_ks_obj.pvalue
sine_exp_ksval = sine_exp_ks_obj.statistic
binom_nll = UnbinnedNLL(col3, binom_pdf)
binom_minuit = Minuit(binom_nll, n=12, p=0.5)
binom_minuit.limits["p"] = (0, 1)
binom_minuit.migrad()
binom_minuit.hesse()

poisson_nll = UnbinnedNLL(col3, poisson_pdf)
poisson_minuit = Minuit(poisson_nll, lam=5)
poisson_minuit.migrad()
poisson_minuit.hesse()

lam_fit = poisson_minuit.values[0]
lam_fit_err = poisson_minuit.errors[0]

p_val_poisson_ks = ks_1samp(col3, poisson(lam_fit).cdf).pvalue


gauss_nll = UnbinnedNLL(col4, gauss_pdf)
gauss_minuit = Minuit(gauss_nll, mu=1.1, sigma=0.05)
gauss_minuit.migrad()
gauss_minuit.hesse()

mu_fit, sigma_fit = gauss_minuit.values[:]
mu_fit_err, sigma_fit_err = gauss_minuit.errors[:]

gauss_ks_obj = ks_1samp(col4, norm(loc=mu_fit, scale=sigma_fit).cdf)
p_val_gauss = gauss_ks_obj.pvalue
ks_val_gauss = gauss_ks_obj.statistic

n_bins_col1 = n_bins_col4 = 50
n_bins_col3 = 19
col3_min, col3_max = (-0.5, 18.5)
col4_min, col4_max = (1, 1.3)
bin_width_col1 = (col1_max - col1_min) / n_bins_col1
bin_width_col3 = 1
bin_width_col4 = (col4_max - col4_min) / n_bins_col4

counts_col3, bin_edges_col3 = np.histogram(
    col3, bins=n_bins_col3, range=(col3_min, col3_max)
)
bin_centers_col3 = (bin_edges_col3[1:] + bin_edges_col3[:-1]) / 2

counts_col3_expected = (
    len(col3) * bin_width_col3 * poisson_pdf(bin_centers_col3, lam_fit)
)
chi2_poisson = np.sum((counts_col3 - counts_col3_expected) ** 2 / counts_col3_expected)

# -1 for 1 parameter and -1 again because normalization loses us a DOF
ndof_poisson = len(counts_col3_expected) - 1 - 1
p_val_poisson_chi2 = chi2.sf(chi2_poisson, ndof_poisson)

sine_latex_strings = [
    rf"$a\approx$ {a_sine_exp_fit:.3f} $\pm$ {a_sine_exp_fit_err:.3f}",
    rf"$b\approx$ {b_sine_exp_fit:.2f} $\pm$ {b_sine_exp_fit_err:.2f}",
    rf"$c\approx$ {c_sine_exp_fit:.0f} $\pm$ {c_sine_exp_fit_err:.0f}",
    rf"$P(KS\approx{sine_exp_ksval:.3f})\approx$ {sine_exp_pval:.3f}",
]
poisson_latex_strings = [
    rf"$\lambda\approx$ {lam_fit:.2f} $\pm$ {lam_fit_err:.2f}",
    rf"$P(\chi^{{2}}\approx{chi2_poisson:.2f},N_{{DOF}}\approx{ndof_poisson:.0f})\approx$ {p_val_poisson_chi2:.2f}",
]
gauss_latex_strings = [
    rf"$\mu\approx$ {mu_fit:.4f} $\pm$ {mu_fit_err:.4f}",
    rf"$\sigma\approx$ {sigma_fit:.4f} $\pm$ {sigma_fit_err:.4f}",
    rf"$P(KS\approx{ks_val_gauss:.3f})\approx$ {p_val_gauss:.3f}",
]
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    col1,
    bins=n_bins_col1,
    ec="k",
    range=(col1_min, col1_max),
    color="deepskyblue",
    label="Data",
)
ax.plot(
    np.sort(col1),
    len(col1)
    * bin_width_col1
    * sine_exp_pdf(
        np.sort(col1),
        a_sine_exp_fit,
        b_sine_exp_fit,
        c_sine_exp_fit,
    ),
    label="sine+exp fit",
    color="Tomato",
    lw=2,
)
ax.text(
    0.95,
    0.5,
    "\n".join(sine_latex_strings),
    transform=ax.transAxes,
    ma="left",
    va="center",
    ha="right",
    fontsize=13,
    family="monospace",
)
ax.legend(fontsize=15, frameon=False)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
plt.savefig("sin_exp.png", dpi=500, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    col3,
    bins=n_bins_col3,
    ec="k",
    range=(col3_min, col3_max),
    color="deepskyblue",
    label="Data",
)
ax.plot(
    np.sort(col3),
    len(col3) * bin_width_col3 * poisson_pdf(np.sort(col3), lam_fit),
    label="Poisson fit",
    color="Tomato",
    lw=2,
)
ax.legend(fontsize=15, frameon=False, loc="lower right")
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
ax.text(
    0.95,
    0.95,
    "\n".join(poisson_latex_strings),
    transform=ax.transAxes,
    ma="left",
    va="top",
    ha="right",
    fontsize=13,
    family="monospace",
)
plt.savefig("poisson.png", dpi=500, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    col4,
    bins=n_bins_col4,
    ec="k",
    range=(col4_min, col4_max),
    color="deepskyblue",
    label="Data",
)
ax.plot(
    np.sort(col4),
    len(col4) * bin_width_col4 * gauss_pdf(np.sort(col4), mu_fit, sigma_fit),
    label="Gauss fit",
    color="Tomato",
    lw=2,
)
ax.text(
    0.98,
    0.7,
    "\n".join(gauss_latex_strings),
    transform=ax.transAxes,
    ma="left",
    va="top",
    ha="right",
    fontsize=13,
    family="monospace",
)
ax.legend(fontsize=15, frameon=False)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
plt.savefig("gauss.png", dpi=500, bbox_inches="tight")
plt.show()

print(f"\nPoisson fit: {lam_fit} +/- {lam_fit_err}")
print(f"\nGauss: {mu_fit} +/- {mu_fit_err}, {sigma_fit} +/- {sigma_fit_err}")


print(
    f"p-values for different fits\n sine+exp: {sine_exp_pval:.3f}\n Poisson: {p_val_poisson_chi2:.3f} \n Gaussian: {p_val_gauss:.3f}"
)
