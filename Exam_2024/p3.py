import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nestle import sample, mean_and_cov
import corner
import scienceplots

plt.style.use("science")


def loglikelihood(theta):
    sigma = np.sqrt(0.04)
    mu = 0.68
    likelihood = 3 * (
        np.cos(theta[0]) * np.cos(theta[1])
        + 1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((theta[2] - mu) ** 2) / (2 * sigma**2))
        * np.cos(theta[0] / 2)
        + 3
    )
    return np.log(likelihood)


def prior_transform(theta):
    return np.array([7 * theta[0] * np.pi, 7 * theta[1] * np.pi, 3 * theta[2]])


res = sample(loglikelihood, prior_transform, ndim=3, method="multi", npoints=10_000)
p, cov = mean_and_cov(res.samples, res.weights)
theta1, theta2, theta3 = p
theta1_err, theta2_err, theta3_err = np.sqrt(np.diag(cov))
print(res.summary())
corner.corner(
    res.samples,
    weights=res.weights,
    range=[
        (0, 7 * np.pi),
        (0, 7 * np.pi),
        (0, 3),
    ],
    show_titles=True,
    labels=[r"$\theta_1$", r"$\theta_2$", r"$\theta_3$"],
)
plt.savefig("corner_p3.png", dpi=500, bbox_inches="tight")
plt.show()

print(
    f"\ntheta1: {theta1} +/- {theta1_err}\ntheta2: {theta2} +/- {theta2_err}\ntheta3: {theta3} +/- {theta3_err}"
)

N_lin = 500
theta1_lin = np.linspace(0, 7 * np.pi, N_lin)
theta2_lin = np.linspace(0, 7 * np.pi, N_lin)
theta3_lin = np.linspace(0, 3, N_lin)
theta1_mesh, theta2_mesh = np.meshgrid(theta1_lin, theta2_lin)
theta1_mesh, theta3_mesh = np.meshgrid(theta1_lin, theta3_lin)


llh_12 = np.array(
    [
        loglikelihood([theta1_mesh.ravel()[i], theta2_mesh.ravel()[i], theta3])
        for i in range(len(theta1_mesh.ravel()))
    ]
).reshape(N_lin, N_lin)
llh_13 = np.array(
    [
        loglikelihood([theta1_mesh.ravel()[i], theta2, theta3_mesh.ravel()[i]])
        for i in range(len(theta1_mesh.ravel()))
    ]
).reshape(N_lin, N_lin)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cf1 = axs[0].contourf(theta1_mesh, theta2_mesh, llh_12, levels=100, cmap="coolwarm")
axs[0].set_xlabel(r"$\theta_{{1}}$", fontsize=17)
axs[1].set_xlabel(r"$\theta_{{1}}$", fontsize=17)
axs[0].set_ylabel(r"$\theta_{{2}}$", fontsize=17)
axs[1].set_ylabel(r"$\theta_{{3}}$", fontsize=17)
cf2 = axs[1].contourf(theta1_mesh, theta3_mesh, llh_13, levels=100, cmap="coolwarm")
fig.colorbar(cf1, ax=axs[0])
fig.colorbar(cf2, ax=axs[1])
plt.savefig("raster_llh_2d.png", dpi=500, bbox_inches="tight")
plt.show()
