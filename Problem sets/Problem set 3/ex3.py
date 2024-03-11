import pandas as pd
import numpy as np
from scipy.special import erf
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import matplotlib.pyplot as plt
from scipy.stats import chi2, ks_1samp
import scienceplots

plt.style.use("science")

data_path = "https://www.nbi.dk/~koskinen/Teaching/data/NucData.txt"
data_vals = pd.read_csv(data_path, names=["data"]).values.T[0]
data_reshaped = data_vals.reshape(100, -1)


def pdf_convolution(t, b, sigma_t):
    return (
        np.exp((sigma_t**2 - 2 * b * t) / (2 * b**2))
        * (erf((b * t - sigma_t**2) / (np.sqrt(2) * b * sigma_t)) + 1)
        / (2 * b)
    )

pdf_convolution_null = lambda t, sigma_t: pdf_convolution(t=t, b=1, sigma_t=sigma_t)
wilk_vals = []
for data in data_reshaped:
    nll_null = UnbinnedNLL(data, pdf_convolution_null)
    nll_alt = UnbinnedNLL(data, pdf_convolution)

    minuit_null = Minuit(nll_null, sigma_t=0.5)
    minuit_alt = Minuit(nll_alt, b=1, sigma_t=0.5)

    minuit_null.migrad()
    minuit_null.hesse()
    minuit_alt.migrad()
    minuit_alt.hesse()
    wilk_vals.append(minuit_null.fval - minuit_alt.fval)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(wilk_vals, bins=50, histtype="step", color="red")
ax.set_xlabel("$-2\ln(\lambda)$", fontsize=17)
ax.set_ylabel("Frequency", fontsize=17)
plt.show()

ndof_wilk = minuit_alt.nfit - minuit_null.nfit
pval_ks = ks_1samp(wilk_vals, chi2.cdf, args=(1,)).pvalue
lower_bound_chi2 = 2.706
n_experiments = (np.array(wilk_vals) > lower_bound_chi2).sum()
survival_prob_lower_bound = chi2.sf(lower_bound_chi2, df=ndof_wilk)
print(f"\np-value KS: {pval_ks}")
print(f"\nNumber of experiments >{lower_bound_chi2}: {n_experiments}")
print(
    f"\nSurvivial probability lower bound: {survival_prob_lower_bound}",
    f"\nThus, we would expect around 10 values >{lower_bound_chi2}"
    )
