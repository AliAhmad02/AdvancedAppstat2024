import pandas as pd
import numpy as np
from scipy.special import erfc
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL
import matplotlib.pyplot as plt
from scipy.stats import chi2, ks_1samp, binom
import scienceplots

plt.style.use("science")

data_path = "https://www.nbi.dk/~koskinen/Teaching/data/NucData.txt"
data_vals = pd.read_csv(data_path, names=["data"]).values.T[0]
data_reshaped = data_vals.reshape(100, -1)


def pdf_convolution(t, b, sigma_t):
    return (
        np.exp((sigma_t**2 - 2 * t * b) / (2 * b**2))
        * erfc((sigma_t**2 - b * t) / (np.sqrt(2) * sigma_t * b))
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
p_vals_chi2 = chi2.sf(wilk_vals, ndof_wilk)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(p_vals_chi2, bins=30, histtype="step", color="red")
ax.set_xlabel("$P(\chi^{{2}}, N_{{DOF}}=1)$", fontsize=17)
ax.set_ylabel("Frequency", fontsize=17)
plt.show()

pval_ks = ks_1samp(wilk_vals, chi2.cdf, args=(1,)).pvalue
lower_bound_chi2 = 2.706
n_experiments = (np.array(wilk_vals) > lower_bound_chi2).sum()
survival_prob_lower_bound = chi2.sf(lower_bound_chi2, df=ndof_wilk)
# k here is exclusive so for 11 or more, we need to plug in 10.
prob_eleven_or_more_events = binom.sf(
    k=n_experiments - 1, n=100, p=survival_prob_lower_bound
)
print(f"\np-value KS: {pval_ks:.2f}")
print(f"\nNumber of experiments >{lower_bound_chi2}: {n_experiments}")
print(
    f"\nSurvivial probability lower bound: {survival_prob_lower_bound:.2f}",
    f"\nThus, we would expect around 10 values >{lower_bound_chi2}",
)
print(f"\nProbability of getting 11 or more events: {prob_eleven_or_more_events:.2f}")
