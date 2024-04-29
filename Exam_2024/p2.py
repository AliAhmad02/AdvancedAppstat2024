import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

data_path = "AdvancedAppstat/Exam_2024/Data_Files/Exam_2024_Problem2.txt"
data = pd.read_csv(data_path, delimiter="\s", engine="python", names=["A", "B"])
data_azimuth, data_zenith = data.T.values


def two_point_autocorr(azimuth, zenith, N_gen_cos):
    N_tot = len(azimuth)
    nx = np.cos(azimuth) * np.sin(zenith)
    ny = np.sin(azimuth) * np.sin(zenith)
    nz = np.cos(zenith)
    cos_gen = np.linspace(-1, 1, N_gen_cos)
    heaviside_vals = np.zeros(len(cos_gen))
    for i in range(N_tot):
        for j in range(i):
            cos_phi_ij = nx[i] * nx[j] + ny[i] * ny[j] + nz[i] * nz[j]
            heaviside_vals += np.heaviside(cos_phi_ij - cos_gen, 0)
    autocorr = 2 / (N_tot * (N_tot - 1)) * heaviside_vals
    return cos_gen, autocorr


def ks(corr1, corr2):
    return np.max(np.abs(corr1 - corr2))


def isotropic_mc(N_gen_mc):
    azimuth = np.random.uniform(0, 2 * np.pi, size=N_gen_mc)
    cos_zenith = np.random.uniform(-1, 1, size=N_gen_mc)
    zenith = np.arccos(cos_zenith)
    return azimuth, zenith


def perform_mc_ks(N_exp, N_gen_mc, N_gen_cos, mc_func):
    ks_vals = []
    for i in range(N_exp):
        print(i)
        azimuth, zenith = mc_func(N_gen_mc)
        cos_gen, tp_autocorr = two_point_autocorr(azimuth, zenith, N_gen_cos)
        iso_autocorr = 1 / 2 * (1 - cos_gen)
        ks_vals.append(ks(tp_autocorr, iso_autocorr))
    return np.array(ks_vals)


def ks_test_isotropic(N_exp, N_gen_mc, N_gen_cos, ks_value, mc_func):
    ks_vals_mc = perform_mc_ks(N_exp, N_gen_mc, N_gen_cos, mc_func)
    p_val = np.sum(ks_vals_mc >= ks_value) / len(ks_vals_mc)
    return ks_vals_mc, p_val


def ha_mc(N_gen_mc):
    N_20_pct = int(np.round(0.2 * N_gen_mc))
    N_80_pct = N_gen_mc - N_20_pct
    azimuth20 = np.random.uniform(0.225 * np.pi, 0.725 * np.pi, size=N_20_pct)
    azimuth80 = np.random.uniform(0, 2 * np.pi, size=N_80_pct)
    cos_zenith20 = np.random.uniform(np.cos(np.pi), np.cos(0.3 * np.pi), size=N_20_pct)
    cos_zenith80 = np.random.uniform(-1, 1, size=N_80_pct)
    azimuth = np.append(azimuth20, azimuth80)
    zenith = np.arccos(np.append(cos_zenith20, cos_zenith80))
    return azimuth, zenith


def hb_mc(N_gen_mc):
    N_15_pct = int(np.round(0.15 * N_gen_mc))
    N_85_pct = N_gen_mc - N_15_pct
    azimuth15 = np.random.uniform(0, np.pi, size=N_15_pct)
    azimuth85 = np.random.uniform(0, 2 * np.pi, size=N_85_pct)
    cos_zenith15 = np.random.uniform(np.cos(np.pi), np.cos(np.pi / 2), size=N_15_pct)
    cos_zenith85 = np.random.uniform(-1, 1, size=N_85_pct)
    azimuth = np.append(azimuth15, azimuth85)
    zenith = np.arccos(np.append(cos_zenith15, cos_zenith85))
    return azimuth, zenith


fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": "mollweide"})
ax.set_title("Raw data", fontsize=20)
ax.grid(color="gray", ls="dotted")
ax.scatter(data_azimuth - np.pi, -data_zenith + np.pi / 2, marker="*", color="black")
plt.savefig("mollweide.png", dpi=500, bbox_inches="tight")
plt.show()

N_gen_cos = 100
cos_gen, tp_autocorr = two_point_autocorr(data_azimuth, data_zenith, N_gen_cos)
iso_autocorr = 1 / 2 * (1 - cos_gen)
ks_val_data = ks(tp_autocorr, iso_autocorr)
N_exp = 3000
N_gen_mc = len(data_zenith)

print(f"\nKS-value: {ks_val_data:.3f}")
ks_vals_mc, p_val_ks = ks_test_isotropic(
    N_exp, N_gen_mc, N_gen_cos, ks_val_data, isotropic_mc
)
ks_vals_mc_ha, p_val_ks_ha = ks_test_isotropic(
    N_exp, N_gen_mc, N_gen_cos, ks_val_data, ha_mc
)
ks_vals_mc_hb, p_val_ks_hb = ks_test_isotropic(
    N_exp, N_gen_mc, N_gen_cos, ks_val_data, hb_mc
)


fig, ax = plt.subplots(1, 1, figsize=(7, 4))
ax.hist(
    ks_vals_mc, ec="k", bins=100, color="deepskyblue", label=f"MC-generated KS-values"
)
ax.axvline(
    ks_val_data, color="black", ls="dashed", label=f"KS-value, data: {ks_val_data:.3f}"
)
ax.set_xlabel("Kolmogorov-Smirnov statistic", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
ax.legend(fontsize=13, frameon=False)
plt.savefig("ks_data_mc.png", dpi=500, bbox_inches="tight")
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].hist(ks_vals_mc_ha, ec="k", bins=100, color="deepskyblue", label="$H_{{A}}$")
axs[0].axvline(
    ks_val_data, color="black", ls="dashed", label=f"$KS_{{data}}$={ks_val_data:.3f}"
)
axs[1].hist(ks_vals_mc_hb, ec="k", bins=100, color="tomato", label="$H_{{B}}$")
axs[1].axvline(
    ks_val_data, color="black", ls="dashed", label=f"$KS_{{data}}$={ks_val_data:.3f}"
)
axs[0].set_xlabel("Kolmogorov-Smirnov statistic", fontsize=15)
axs[0].set_ylabel("Frequency", fontsize=15)
axs[0].legend(fontsize=13, frameon=False)
axs[1].set_xlabel("Kolmogorov-Smirnov statistic", fontsize=15)
axs[1].set_ylabel("Frequency", fontsize=15)
axs[1].legend(fontsize=13, frameon=False)
plt.savefig("ks_data_mc_hahb.png", dpi=500, bbox_inches="tight")
plt.show()


print(f"\nKS p-value: {p_val_ks}")
print(f"\nKS p-value, HA: {p_val_ks_ha}")
print(f"\nKS p-value, HB: {p_val_ks_hb}")
