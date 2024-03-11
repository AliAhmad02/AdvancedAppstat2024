import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from datetime import datetime
import sympy as sy
from sklearn.neighbors import KernelDensity

plt.style.use("science")

crash_path = (
    "AdvancedAppstat/Problem sets/Problem set 3/data_files/cpd-crash-incidents.csv"
)
crash_df = pd.read_csv(crash_path, delimiter=";", usecols=["lon", "lat", "crash_date"])
crash_times = crash_df["crash_date"].str.split("T").str[-1].str.split("+").str[0]
crash_hours_int = crash_times.str.split(":").str[0].astype("int").values
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(
    crash_df["lon"].values, crash_df["lat"].values, ".", markersize=5, color="black"
)
ax.set_xlabel("Longitude (decimal degrees)")
ax.set_ylabel("Latitude (decimal degrees)")
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(crash_hours_int, bins=24, range=(0, 24), color="black", histtype="step")
ax.set_xlabel("Crash time [hr]", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
ax.set_xticks(range(0, 25))
plt.show()


class EpanechnikovKernelTimeDiff:
    def __init__(self, data, h):
        self.data = data
        self.h = h
        self.kernel = self._get_kernel_func()

    def _get_kernel_func(self):
        u = sy.Symbol("u", real=True)
        height = 3 / 4 * (1 - u**2)
        height_norm = height / self.h
        kernel = sy.Piecewise((height_norm, sy.Abs(u) <= 1), (0, sy.Abs(u) > 1))
        kernel_func = sy.lambdify(u, kernel)
        return kernel_func

    def __call__(self, time):
        timediffs1 = (time - self.data).seconds
        timediffs2 = (self.data - time).seconds
        timediffs_df = pd.DataFrame({"diff1": timediffs1, "diff2": timediffs2})
        timediff_hrs = timediffs_df.min(axis=1).values / (60 * 60)
        u = timediff_hrs / self.h
        return self.kernel(u)


times_linspace = pd.date_range(
    "01-01-2024 00:00:00", "01-01-2024 23:59:00", freq="1min"
)
crash_times_dt = pd.to_datetime(
    np.char.add("01-01-2024 ", crash_times.values.astype("str"))
)
times_linspace_num = np.linspace(0, 24 - 1 / 60, len(times_linspace))
bw_epan = 0.8
kernel_epan = EpanechnikovKernelTimeDiff(crash_times_dt, bw_epan)
kernel_vals_lin = [kernel_epan(time) for time in times_linspace]
kde_epan = 1 / len(crash_times) * np.sum(kernel_vals_lin, axis=1)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(times_linspace_num, kde_epan, color="red", label="KDE crash times")
ax.set_xlabel("Time [hours]", fontsize=15)
ax.hist(
    crash_hours_int,
    bins=24,
    range=(0, 24),
    color="black",
    histtype="step",
    density=True,
    label="Density crash hours",
)
ax.set_ylabel("Probability density", fontsize=15)
ax.set_xticks(range(0, 25))
ax.legend(fontsize=15, frameon=False)
plt.show()

times_eval = [
    "01-01-2024 00:23:00",
    "01-01-2024 01:49:00",
    "01-01-2024 08:12:00",
    "01-01-2024 15:55:00",
    "01-01-2024 18:02:00",
    "01-01-2024 21:12:00",
    "01-01-2024 23:44:00",
]
prob_times_eval = kde_epan[times_linspace.isin(times_eval)]
print(f"\nProbabilities at the suggested times: {prob_times_eval}")

crash_areas = []
max_time = times_linspace_num[-1] - 2
times_scan = times_linspace_num[times_linspace_num < max_time]
for time in times_scan:
    window_mask = (times_linspace_num >= time) & (times_linspace_num <= (time + 2))
    time_window = times_linspace_num[window_mask]
    kde_window = kde_epan[window_mask]
    crash_areas.append(np.trapz(kde_window, time_window))
max_prob_time = times_scan[np.argmax(crash_areas)]
max_prob = np.amax(crash_areas)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(times_scan, crash_areas, color="red", label="Crash probability")
ax.set_xlabel("Patrol time start [hrs]", fontsize=15)
ax.set_ylabel("2-hour crash probability", fontsize=15)
ax.axvline(
    max_prob_time,
    color="black",
    linestyle="dashed",
    lw=1.5,
    label="Maximum crash probability",
)
ax.text(
    max_prob_time,
    0.08,
    "$t_{{max}}=$19:58",
    fontsize=12,
    bbox=dict(fc="white", edgecolor="black"),
    ha="center",
)
ax.legend(fontsize=15, frameon=False)
plt.show()

print(f"\nPercentage fewer crashes due to 2-hour patrol: {max_prob*0.1*100:.2f}%")


# Function courtesy of user: https://stackoverflow.com/users/7415588/geoffrey
def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    xx, yy = np.mgrid[x.min() : x.max() : xbins, y.min() : y.max() : ybins]
    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


bw_long_lat = 0.01
long_mesh, lat_mesh, long_lat_kde = kde2D(
    crash_df["lon"].dropna().values,
    crash_df["lat"].dropna().values,
    bw_long_lat,
    kernel="epanechnikov",
)
plt.figure(figsize=(6, 4))
plt.contourf(long_mesh, lat_mesh, long_lat_kde, levels=100, cmap="hot")
plt.xlabel("Longitude (decimal degrees)", fontsize=15)
plt.ylabel("Latitude (decimal degrees)", fontsize=15)
cbar = plt.colorbar()
cbar.set_label("Probability density", fontsize=15)
plt.show()

lat_box_lower, lat_box_upper = 35.74, 35.78
long_box_lower, long_box_upper = -78.76, -78.72

lat_mask = (lat_mesh[0] >= lat_box_lower) & (lat_mesh[0] <= lat_box_upper)
long_mask = (long_mesh[:, 0] >= long_box_lower) & (long_mesh[:, 0] <= long_box_upper)
lat_stacked = np.vstack([lat_mask] * len(long_lat_kde))
long_stacked = np.column_stack([long_mask] * len(long_lat_kde))
tot_mask = lat_stacked * long_stacked
integral_2d = np.trapz(np.trapz(long_lat_kde * tot_mask, lat_mesh[0]), long_mesh[:, 0])
print(f"\nArea inside box: {integral_2d:.2f}")
