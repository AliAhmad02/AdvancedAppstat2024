"""General functions useful for solving problems in the course."""

import pandas as pd
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import inspect
from sympy import symbols, integrate, lambdify
from iminuit import Minuit
from iminuit.cost import UnbinnedNLL


def normalize_pdf_1d(
    pdf: Callable[..., NDArray],
    lower: float,
    upper: float,
) -> Callable[..., NDArray]:
    """Convert a function into a normalized PDF.

    Assumes that the function "pdf" takes in the independent
    variable as the first parameter.

    Args:
        pdf: The non-normalized function we want to use as a PDF.
        lower: Lower bound on the domain of the pdf.
        upper: Upper bound on the domain of the pdf.

    Return:
        norm_pdf_func: Normalized PDF.
    """
    # Get list of parameters that the pdf takes.
    pdf_params = list(inspect.signature(pdf).parameters.keys())
    # Turn the parameters into sympy symbols
    pdf_params_symbs = symbols(",".join(pdf_params))
    # Turn our function into a sympy expression
    pdf_symp = pdf(*pdf_params_symbs)
    # Calculate the integral and thus, the normalization
    integral = integrate(pdf_symp, (pdf_params_symbs[0], lower, upper))
    norm_pdf_symp = pdf_symp / integral
    # Turn the sympy expresson for the normalized pdf into a Python function
    norm_pdf_func = lambdify(pdf_params_symbs, norm_pdf_symp)
    return norm_pdf_func


def raster_scan_llh_2d(
    data: NDArray,
    pdf: Callable[..., NDArray],
    par1_range: tuple[float, float],
    par2_range: tuple[float, float],
    N: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Raster scan over the log-likelihood for different values of two parameters.

    Args:
        data: The data for which to calculate the log-likelihood.
        pdf: The PDF. Should take arguments (data, par1, par2)
        par1_range: The range over which we scan for the first parameter.
        par2_range: The range over which we scan for the first parameter.
        N: Number of values to scan over for each parameter.

    Return:
        par1_mesh: A 2D mesh of par1 values.
        par2_mesh: A 2D mesh of par2 values.
        llh_pdf_mesh: A 2D mesh of log-likelihood values.
        max_llh_par1: Maximum likelihood estimate of par1.
        max_llh_par2: Maximum likelihood estimate of par2.
    """
    # Define that values for the parameters we want to scan over.
    par1_values = np.linspace(*par1_range, N)
    par2_values = np.linspace(*par2_range, N)
    # Create meshgrid of all possible parameter combinations.
    par1_mesh, par2_mesh = np.meshgrid(par1_values, par2_values)
    # Create a 2D array of all possible pdf values
    pdf_mesh = pdf(
        data.reshape(1, -1),
        par1_mesh.reshape(-1, 1),
        par2_mesh.reshape(-1, 1),
    )
    # Create a 2D array of all possible likelihood values
    llh_pdf_mesh = np.sum(np.log(pdf_mesh), axis=1).reshape(N, N)
    # Find the index of the maximum likelihood value
    max_llh_idx = np.argmax(llh_pdf_mesh.ravel())
    # Evaluate the parameters at the maximum likelihood value
    max_llh_par1 = par1_mesh.ravel()[max_llh_idx]
    max_llh_par2 = par2_mesh.ravel()[max_llh_idx]
    return par1_mesh, par2_mesh, llh_pdf_mesh, max_llh_par1, max_llh_par2


def bootstrap_llh(
    data: NDArray,
    pdf: Callable[..., NDArray],
    lower: float,
    upper: float,
    pmax: float,
    N_boot: int,
    N_gen: int,
    **kwargs,
) -> pd.DataFrame:
    """Perform bootstrapping for parameter estimation.

    Args:
        data: The data sampled from a certain pdf.
        pdf: The pdf in question.
        lower: Lower bound on the domain of the pdf.
        upper: Upper bound on the domain of the pdf.
        pmax: The maximum probability of the pdf. Can
        be set to 1 for simplicity, at the cost efficiency.
        N_boot: Number of times to perform the fits.
        N_gen: Number of points to sample from the pdf each time.
        kwargs: Initial guesses for the fitting parameters.

    Return:
        params_df: All N_boot estimates of all parameters.
    """
    # Instantiate object for -2*LLH (cost function to be minimized)
    nll_obj_init = UnbinnedNLL(data, pdf)
    # Instantiate Minuit object
    minuit_obj_init = Minuit(nll_obj_init, **kwargs)
    # Perform the fit
    minuit_obj_init.migrad()
    minuit_obj_init.hesse()
    # Get parameter key-value pairs from the first fit
    pars_init = minuit_obj_init.values.to_dict()
    # Initialize empty list to store bootstrap parameter values.
    pars_boot = []
    for i in range(N_boot):
        # Sample from the PDF using accept-reject.
        x = np.random.uniform(lower, upper, N_gen)
        y = np.random.uniform(0, pmax, N_gen)
        f_x = pdf(x, **pars_init)
        x_accepted = x[y < f_x]
        # Perform fit to sample, using pars_init as initial guess for fit.
        nll_obj_boot = UnbinnedNLL(x_accepted, pdf)
        minuit_obj_boot = Minuit(nll_obj_boot, **pars_init)
        minuit_obj_boot.migrad()
        minuit_obj_boot.hesse()
        pars_boot.append(minuit_obj_boot.values.to_dict())
    params_df = pd.DataFrame(pars_boot)
    return params_df
