import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


# Requires scienceplots style installation
import scienceplots

plt.style.use(["science"])


x_min = 0
x_mid = 1.6
x_max = 2.1
N_lin_prior = 1000

lower = 0
upper = 4
x = np.linspace(lower, upper, N_lin_prior)
norm_prior = 2.93
prior = np.piecewise(
    x,
    [
        (x > x_min) & (x <= x_mid),
        (x > x_mid) & (x <= x_max),
        (x <= x_min) | (x > x_max),
    ],
    [1 / norm_prior, 2.66 / norm_prior, 0],
)

alpha = 2
beta = 2
likelihood = gamma.pdf(x, a=alpha, scale=1 / beta)
posterior = prior * likelihood / np.trapz(prior * likelihood, x=x)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(x, prior, label=f"Prior", color="black")
ax.plot(x, likelihood, label=f"Likelihood", color="darkred")
ax.plot(
    x,
    posterior,
    label=f"Posterior",
    color="deeppink",
)
ax.set_xlabel("x", fontsize=15)
ax.set_ylabel("Probability", fontsize=15)
plt.legend(frameon=False, fontsize=13)
plt.show()

max_posterior_idx = np.argmax(posterior)
max_posterior_x = x[max_posterior_idx]
max_posterior_val = posterior[max_posterior_idx]

print(f"\nThe most like estimate is g(x={max_posterior_x:.2f})={max_posterior_val:.2f}")
