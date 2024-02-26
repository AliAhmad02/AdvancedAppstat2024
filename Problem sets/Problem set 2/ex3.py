import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix

# Requires scienceplots style installation
import scienceplots

plt.style.use(["science"])


def get_distance_to_border(x, y, angle, r):
    alpha = x * np.cos(angle) + y * np.sin(angle)
    return -alpha + np.sqrt(alpha**2 - x**2 - y**2 + r**2)


def one_crab_walk(x_start, y_start, max_step_size, N_days, radius):
    x = x_start
    y = y_start
    x_vals = [x_start]
    y_vals = [y_start]
    for i in range(N_days):
        rand_angle = np.random.uniform(0, 2 * np.pi)
        dist_rand_angle = get_distance_to_border(x, y, rand_angle, radius)
        if dist_rand_angle > max_step_size:
            x += max_step_size * np.cos(rand_angle)
            y += max_step_size * np.sin(rand_angle)
        else:
            x += dist_rand_angle * np.cos(rand_angle)
            y += dist_rand_angle * np.sin(rand_angle)
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals


def one_crab_walk_to_edge(x_start, y_start, max_step_size, radius):
    x = x_start
    y = y_start
    total_distance = 0
    while True:
        rand_angle = np.random.uniform(0, 2 * np.pi)
        dist_rand_angle = get_distance_to_border(x, y, rand_angle, radius)
        if dist_rand_angle > max_step_size:
            x += max_step_size * np.cos(rand_angle)
            y += max_step_size * np.sin(rand_angle)
            total_distance += max_step_size
        else:
            x += dist_rand_angle * np.cos(rand_angle)
            y += dist_rand_angle * np.sin(rand_angle)
            total_distance += dist_rand_angle
            break
    return total_distance


x_start = 3.6
y_start = -2.0
N_days = 200
max_step_size = 0.2
radius = 5

r = 5
N = 100_000
x = np.random.uniform(-r, r, N)
y = np.random.uniform(-r, r, N)
circ_mask = (x**2 + y**2) <= r**2
x = x[circ_mask]
y = y[circ_mask]

x_one_crab, y_one_crab = one_crab_walk(x_start, y_start, max_step_size, N_days, radius)
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_facecolor("aqua")
ax.scatter(x, y, color="wheat", label="Island", s=5, zorder=0)
ax.scatter(
    x_one_crab,
    y_one_crab,
    c=np.arange(len(x_one_crab)),
    cmap="Greys",
    s=5,
    label="Moves",
)
ax.set_xlabel(r"$x$ [km]", fontsize=17)
ax.set_ylabel(r"$y$ [km]", fontsize=17)
ax.legend(fontsize=13, frameon=False, markerscale=2.5)
plt.show()

N_experiments = 501
distances_edge_walk = [
    one_crab_walk_to_edge(x_start, y_start, max_step_size, radius)
    for i in range(N_experiments)
]
edge_walk_min = 0
edge_walk_max = 200
bin_width_edge_walk = 2
n_bins_edge_walk = int((edge_walk_max - edge_walk_min) / bin_width_edge_walk)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    distances_edge_walk,
    bins=n_bins_edge_walk,
    range=(edge_walk_min, edge_walk_max),
    histtype="stepfilled",
    color="hotpink",
)
ax.set_xlabel("Total distance traveled [km]", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
plt.show()

crab_positions_path = (
    "https://www.nbi.dk/~koskinen/Teaching/data/CrabStartPositions.txt"
)
crab_positions_x, crab_positions_y = pd.read_csv(
    crab_positions_path, delimiter=" ", names=["x", "y"]
).T.values


def update_crabs_positions(xs, ys, max_step_size, radius):
    N_crabs = len(xs)
    rand_angles = np.random.uniform(0, 2 * np.pi, N_crabs)
    dists_rand_angles = get_distance_to_border(xs, ys, rand_angles, radius)
    mask_max = dists_rand_angles > max_step_size
    # Update positions where we take the maximum step size
    xs[mask_max] += max_step_size * np.cos(rand_angles[mask_max])
    ys[mask_max] += max_step_size * np.sin(rand_angles[mask_max])
    # Update positions where we take the step size that brings us to the border
    xs[~mask_max] += dists_rand_angles[~mask_max] * np.cos(rand_angles[~mask_max])
    ys[~mask_max] += dists_rand_angles[~mask_max] * np.sin(rand_angles[~mask_max])
    return xs, ys


def get_crabs_within_battle_distance(xs, ys, battle_dist):
    # Create an array where each point is a pair of x-y coordinates
    points_2d_arr = np.vstack((xs, ys)).T
    # Create a matrix with all possible distances calculated
    distances = distance_matrix(points_2d_arr, points_2d_arr)
    # The above matrix has each distance twice, i.e. the distance at index i,j is
    # the same as j,i. Additionally, the diagonal of the matrix is just zero, as
    # it is the distance between a point and itself. Therefore, since we are
    # only interested, uniquely, in the distance between every pair of points, we
    # only consider the triangular part of the matrix above the main diagonal.
    # np.triu_indices returns a tuple of two arrays, the first array containing the
    # row-indices of the elements in the upper triangle and the second containing
    # the column-indices of the elements in the upper triangle of the matrix.
    unique_dist_indices = np.vstack(np.triu_indices(len(points_2d_arr), k=1)).T
    # Get a list of distances between each point (unique)
    distances_unique = distances[unique_dist_indices[:, 0], unique_dist_indices[:, 1]]
    # Get the indices that sort the above distance array. We are interested in
    # this because we want the battles to happen in order of smallest distance
    distances_unique_sort_indices = np.argsort(distances_unique)
    # Sort the indices of all unique pairs according to distance
    unique_dist_indices = unique_dist_indices[distances_unique_sort_indices]
    # Create a boolean mask of the distances that are within battle distance
    battle_range_mask = distances_unique <= battle_dist
    return battle_range_mask, unique_dist_indices


def simulate_crab_battle(battle_pair, crab_masses):
    # if the two crabs have the same mass, arbitrarily assign them big/small
    if crab_masses[battle_pair[0]] == crab_masses[battle_pair[1]]:
        crab_big = battle_pair[0]
        crab_small = battle_pair[1]
    # else assign big/small status according to the mass
    else:
        crab_big = battle_pair[np.argmax(crab_masses[battle_pair])]
        crab_small = battle_pair[np.argmin(crab_masses[battle_pair])]
    # Get the mass of the two crabs
    crab_big_mass = crab_masses[crab_big]
    crab_small_mass = crab_masses[crab_small]
    # Calculate the probability that the small crab gets eaten
    prob_eat = crab_big_mass**2 / (crab_big_mass**2 + crab_small_mass**2)
    # Simulate the crab being eaten with the given probability
    rand_uniform = np.random.uniform(0, 1)
    if prob_eat >= rand_uniform:
        crab_masses[crab_big] += crab_masses[crab_small]
        crab_masses[crab_small] = 0
    return crab_masses


def multiple_crabs_walk(xs_start, ys_start, max_step_size, N_days, radius):
    xs = xs_start
    ys = ys_start
    crab_masses = np.ones(len(xs))
    battle_dist = 0.175
    for i in range(N_days):
        if crab_masses.sum() != len(xs):
            raise ValueError("Crab masses must sum to 20 kg")
        xs, ys = update_crabs_positions(xs, ys, max_step_size, radius)
        battle_range_mask, unique_dist_indices = get_crabs_within_battle_distance(
            xs, ys, battle_dist
        )
        # Move on to the next iteration if no crabs are in battle range
        if battle_range_mask.sum() == 0:
            continue
        # Get the indices of the crabs which are within battle distance of each other
        indices_battle_range = unique_dist_indices[battle_range_mask]
        # Loop all all two-crab battles
        for battle_pair in indices_battle_range:
            # Since we do not remove eaten crabs from our xs and ys arrays,
            # they will still show up in the indices_battle_range array. Thus,
            # we make sure to continue on to the next iteration (meaning we do not
            # start a battle) if a pair of crabs has any crabs that are already eaten.
            # And we check if they are already eaten by checking if the mass is zero.
            if (crab_masses[battle_pair] == 0).any():
                continue
            crab_masses = simulate_crab_battle(battle_pair, crab_masses)
    surviving_crabs = np.arange(len(xs))[crab_masses != 0]
    surviving_crabs_masses = crab_masses[surviving_crabs]
    return surviving_crabs, surviving_crabs_masses


def multiple_crabs_walk_ten_surviving(xs_start, ys_start, max_step_size, radius):
    xs = xs_start
    ys = ys_start
    crab_masses = np.ones(len(xs))
    battle_dist = 0.175
    days = 0
    n_surviving_crabs = len(xs)
    while n_surviving_crabs > 10:
        days += 1
        if crab_masses.sum() != len(xs):
            raise ValueError("Crab masses must sum to 20 kg")
        xs, ys = update_crabs_positions(xs, ys, max_step_size, radius)
        battle_range_mask, unique_dist_indices = get_crabs_within_battle_distance(
            xs, ys, battle_dist
        )
        if battle_range_mask.sum() == 0:
            continue
        indices_battle_range = unique_dist_indices[battle_range_mask]
        for battle_pair in indices_battle_range:
            if (crab_masses[battle_pair] == 0).any():
                continue
            crab_masses = simulate_crab_battle(battle_pair, crab_masses)
            n_surviving_crabs = len(crab_masses[crab_masses != 0])
            if n_surviving_crabs == 10:
                break
    return days


N_surviving_crabs = []
largest_masses = []
days_until_ten_surviving = []
N_experiments1 = 2000

for i in range(N_experiments1):
    if i % 100 == 0:
        print(f"Iteration {i}")
    surviving_crabs, surviving_crabs_masses = multiple_crabs_walk(
        crab_positions_x, crab_positions_y, max_step_size, N_days, radius
    )
    days_until_ten_surviving.append(
        multiple_crabs_walk_ten_surviving(
            crab_positions_x, crab_positions_y, max_step_size, radius
        )
    )
    N_surviving_crabs.append(len(surviving_crabs))
    largest_masses.append(np.max(surviving_crabs_masses))

xmin_surviving_crabs = 2
xmax_surviving_crabs = 16
n_bins_surviving_crabs = xmax_surviving_crabs - xmin_surviving_crabs + 1

xmin_largest_masses = 2
xmax_largest_masses = 14
n_bins_largest_masses = xmax_largest_masses - xmin_largest_masses + 1

xmin_days = 15
xmax_days = 420
n_bins_days = 50

onesig_lower_pct = (100 - 68.27) / 2
onesig_upper_pct = 100 - onesig_lower_pct
onesig_ci_lower_days = np.percentile(days_until_ten_surviving, onesig_lower_pct)
onesig_ci_upper_days = np.percentile(days_until_ten_surviving, onesig_upper_pct)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    N_surviving_crabs,
    bins=n_bins_surviving_crabs,
    color="hotpink",
    range=(xmin_surviving_crabs - 0.5, xmax_surviving_crabs + 0.5),
)
ax.set_xticks(np.arange(xmin_surviving_crabs, xmax_surviving_crabs + 1))
ax.set_xlabel("Number of surviving crabs", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.hist(
    largest_masses,
    bins=n_bins_largest_masses,
    color="hotpink",
    range=(xmin_largest_masses - 0.5, xmax_largest_masses + 0.5),
)
ax.set_xticks(np.arange(xmin_largest_masses, xmax_largest_masses + 1))
ax.set_xlabel("Largest surviving crab mass [kg]", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(
    days_until_ten_surviving,
    bins=n_bins_days,
    color="hotpink",
    range=(xmin_days - 0.5, xmax_days + 0.5),
    label="Histogram",
)
ax.axvspan(
    onesig_ci_lower_days,
    onesig_ci_upper_days,
    color="black",
    alpha=0.2,
    label=f"$1\sigma$ CI [{onesig_ci_lower_days:.1f}, {onesig_ci_upper_days:.1f}]",
)
ax.set_xlabel("Days until 10 surviving crabs", fontsize=15)
ax.set_ylabel("Frequency", fontsize=15)
ax.legend(frameon=False, fontsize=13)
plt.show()
