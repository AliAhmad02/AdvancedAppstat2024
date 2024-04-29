import numpy as np
from scipy.interpolate import CubicSpline, splrep, BSpline
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import scienceplots

plt.style.use("science")

data = np.array(
    [
        [203.41, -89.37],
        [203.435, -94.88],
        [203.46, -101.25],
        [203.484, -106.52],
        [203.509, -108.66],
        [203.534, -114.25],
        [203.558, -114.30],
        [203.583, -117.66],
        [203.608, -122.45],
    ]
)
sol, temp = data.T
temp_err = 0.001
N_interp = 10_000
step_size = 0.0004
spline_linspace = np.arange(203.410, 203.608, step_size)
val = 203.570
linear_spline_val = np.interp(val, sol, temp)
linear_spline_lin = np.interp(spline_linspace, sol, temp)
cubic_spline_obj = CubicSpline(sol, temp)
cubic_spline_val = cubic_spline_obj(val)
cubic_spline_lin = cubic_spline_obj(spline_linspace)
print(
    f"\nTemperature on sol 203.570:\n Linear spline: {linear_spline_val:.3f}\n Cubic spline: {cubic_spline_val:.3f}"
)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(sol, temp, "o", color="black", label="Data")
ax.plot(spline_linspace, linear_spline_lin, color="blue", label="Linear spline", lw=2)
ax.plot(
    spline_linspace,
    cubic_spline_lin,
    "--",
    color="tomato",
    label="Cubic spline",
    lw=2.5,
)
ax.set_xlabel("Sol", fontsize=15)
ax.set_ylabel("Temperature [$^{{o}}C$]", fontsize=15)
ax.legend(fontsize=15, frameon=False)
# plt.savefig("splines1.png", dpi=500, bbox_inches="tight")
plt.show()

smoothing = len(sol) - np.sqrt(2 * len(sol))
tck = splrep(sol, temp, s=smoothing, k=3)
cubic_spline_smooth_lin = BSpline(*tck)(spline_linspace)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(sol, temp, "o", color="black", label="Data")
ax.plot(spline_linspace, cubic_spline_lin, color="blue", label="Cubic spline", lw=2)
ax.plot(
    spline_linspace,
    cubic_spline_smooth_lin,
    color="tomato",
    label="Cubic spline (smoothed)",
    lw=2,
)
ax.set_xlabel("Sol", fontsize=15)
ax.set_ylabel("Temperature [$^{{o}}C$]", fontsize=15)
ax.legend(fontsize=15, frameon=False, loc="lower left")
axins = zoomed_inset_axes(ax, zoom=1.7, loc=1)
axins.plot(spline_linspace, cubic_spline_lin, color="blue", lw=2)
axins.plot(spline_linspace, cubic_spline_smooth_lin, color="tomato", lw=2)
axins.plot(sol, temp, "o", color="black")
axins.set_xlim(203.534, 203.583)
axins.set_ylim(-120, -110)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
plt.savefig("splines2.png", dpi=500, bbox_inches="tight")
plt.show()

max_diff_lin_spline = np.abs(np.diff(linear_spline_lin)).max()
max_diff_cubic_spline = np.abs(np.diff(cubic_spline_lin)).max()
max_diff_cubic_spline_smoothed = np.abs(np.diff(cubic_spline_smooth_lin)).max()

print(
    f"\nMaximum difference within 0.0004 sol:\n Linear spline: {max_diff_lin_spline:.3f}\n Cubic spline: {max_diff_cubic_spline:.3f}\n Cubic spline (smoothed): {max_diff_cubic_spline_smoothed:.3f}"
)
