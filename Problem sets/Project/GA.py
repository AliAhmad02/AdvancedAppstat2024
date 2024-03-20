import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Callable
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec, dblquad
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL


def generate_initial_population(
    bounds_lower: NDArray, bounds_upper: NDArray, n_pop: int
) -> NDArray:
    n_params = len(bounds_upper)
    rand_pop = np.array(
        [
            np.random.uniform(bounds_lower[i], bounds_upper[i], size=n_pop)
            for i in range(n_params)
        ]
    ).T
    return rand_pop


def select_parents(
    population: NDArray, fitness: NDArray[np.float64], n_pairs
) -> NDArray[np.int64]:
    weights = np.abs(np.min(fitness)) + fitness
    parent_pairs = np.array(
        [
            np.random.choice(
                np.arange(0, population.shape[0]),
                size=2,
                replace=False,
                # p=(1-(fitness / np.sum(fitness)))/np.sum((1-(fitness / np.sum(fitness)))),
                p=weights / np.sum(weights),
            )
            for _ in range(n_pairs)
        ]
    )
    return parent_pairs


def one_point_crossover(
    population: NDArray,
    parent_pairs: NDArray,
) -> NDArray:
    n_pairs = len(parent_pairs)
    n_cols = population.shape[1]
    rand_param: NDArray = np.random.randint(1, n_cols, n_pairs)
    pop_vals = population.copy()
    pop_pairs = pop_vals[parent_pairs, :]
    mask = (rand_param.reshape(1, -1) <= np.arange(n_cols).reshape(-1, 1, 1)).T
    pop_pairs = (pop_pairs * ~mask) + (pop_pairs * mask)[:, ::-1, :]
    new_generation = pop_pairs.reshape(2 * n_pairs, -1)
    return new_generation


def uniform_crossover(
    population: NDArray,
    parent_pairs: NDArray,
) -> NDArray:
    n_pairs = len(parent_pairs)
    n_cols = population.shape[1]
    mask = np.random.choice([False, True], size=(n_pairs, n_cols))[:, np.newaxis, :]
    pop_vals = population.copy()
    pop_pairs = pop_vals[parent_pairs, :]
    pop_pairs = pop_pairs * mask + (pop_pairs * ~mask)[:, ::-1, :]
    new_generation = pop_pairs.reshape(2 * n_pairs, -1)
    return new_generation


def mutation(
    population: NDArray,
    bounds_upper: NDArray,
    p_mut: float,
    iteration: int,
    n_generations: int,
) -> NDArray:
    init_widths = bounds_upper / 4
    delta_widths = init_widths / n_generations
    widths = init_widths - iteration * delta_widths
    for individual, row in enumerate(population):
        mut_nums = np.array([np.random.normal(loc=0, scale=width) for width in widths])
        mut_probs = np.random.uniform(low=0, high=1, size=len(widths))
        mask = mut_probs <= p_mut
        if mask.sum() != 0:
            row[mask] += mut_nums[mask]
        population[individual] = row
    return population


def perform_evolution(
    bounds_lower: NDArray,
    bounds_upper: NDArray,
    n_pop: int,
    n_generations: int,
    fitness: Callable,
    p_mut: float,
    n_elite_pairs: int,
    **kwargs,
) -> NDArray:
    population = generate_initial_population(bounds_lower, bounds_upper, n_pop)
    pop_max_fitness_global = []
    max_fitness_global = -999_9999
    for i in range(n_generations):
        fitness_values = fitness(population, **kwargs)
        top_parents_ids = np.argsort(fitness_values)[-int(2 * n_elite_pairs) :]
        top_parents = population[top_parents_ids]
        n_pairs = int(n_pop / 2) - n_elite_pairs
        parents = select_parents(population, fitness_values, n_pairs)
        # new_generation = one_point_crossover(population, parents)
        new_generation = uniform_crossover(population, parents)
        new_generation = np.append(new_generation, top_parents, axis=0)
        new_generation = mutation(new_generation, bounds_upper, p_mut, i, n_generations)
        print(f"Maximum fitness: {np.max(fitness_values)}")
        max_fitness_local = np.max(fitness_values)
        if max_fitness_local > max_fitness_global:
            max_fitness_idx = np.argmax(fitness_values)
            max_fitness_global = fitness_values[max_fitness_idx]
            pop_max_fitness_global = population[max_fitness_idx].copy()
        population = new_generation
    return max_fitness_global, pop_max_fitness_global


def voigt_pdf1(lam, lam_0, gamma, b):
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


def generate_data_voigt_pdf(
    lam_0: float,
    gamma: float,
    b: float,
    lower: int = 6300,
    upper: int = 6400,
    N_gen: int = 200_000,
    pmax: int = 0.4,
):
    x = np.random.uniform(lower, upper, N_gen)
    y = np.random.uniform(0, pmax, N_gen)
    f_x = voigt_pdf1(x, lam_0, gamma, b)
    x_accepted = x[y < f_x]
    return x_accepted


def fitness_llh(population, data):
    pdf_vals = np.array([voigt_pdf1(data, *row) for row in population])
    return 2 * np.sum(np.log(pdf_vals), axis=1)


bounds_lower_llh = np.array([5000, 0, 10])
bounds_upper_llh = np.array([7000, 3, 30])
n_pop_llh = 40
n_generations_llh = 200
p_mut_llh = 0.4
n_elite_pairs_llh = 1
data_with_noise = np.genfromtxt(
    "AdvancedAppstat/Problem sets/Project/Data Files/MC_data.txt", delimiter="\n"
)
nll_gauss_noise = UnbinnedNLL(data_with_noise, voigt_pdf1)
minuit_gauss_noise = Minuit(nll_gauss_noise, *generate_initial_population(bounds_lower_llh, bounds_upper_llh, 1)[0])
minuit_gauss_noise = Minuit(
    nll_gauss_noise, *[6.34996375e03, 6.34554601e-01, 9.46659796e01]
)
minuit_gauss_noise.limits["lam_0"] = (5000, 7000)
minuit_gauss_noise.limits["gamma"] = (0, 3)
minuit_gauss_noise.limits["b"] = (10, 30)
minuit_gauss_noise.migrad()
print(minuit_gauss_noise)
print(-minuit_gauss_noise.fval)
print(2 * np.sum(np.log(voigt_pdf1(data_with_noise, 6350, 1, 17))))
print(perform_evolution(
    bounds_lower_llh,
    bounds_upper_llh,
    n_pop_llh,
    n_generations_llh,
    fitness_llh,
    p_mut_llh,
    n_elite_pairs_llh,
    data=data_with_noise,
))
