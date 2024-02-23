"""General functions useful for solving problems in the course."""

import numpy as np
from typing import Callable
from numpy.typing import NDArray
import inspect
from sympy import symbols, integrate, lambdify


def get_normalization_1d_pdf(
    pdf: Callable[..., NDArray],
    lower: float,
    upper: float,
) -> Callable[..., float]:
    """Get the function that normalizes a PDF.

    Args:
        pdf: The non-normalized function we want to use as a PDF.
        lower: Lower bound on the range of the pdf.
        upper: Upper bound on the range of the pdf.

    Return:
        norm_func: Function for calculating the normalization constant.
    """
    pdf_params = list(inspect.signature(pdf).parameters.keys())
    pdf_params_symbs = symbols(",".join(pdf_params))
    integral = integrate(pdf(*pdf_params_symbs), (pdf_params_symbs[0], lower, upper))
    norm = 1 / integral
    norm_func = lambdify(pdf_params_symbs, norm)
    return norm_func


def normalize_pdf(
    pdf: Callable[..., NDArray],
    lower: float,
    upper: float,
) -> Callable[..., NDArray]:
    """
    Get a normalized PDF using get_normalization_1d_pdf.

    Args:
        pdf: The non-normalized function we want to use as a PDF.
        lower: Lower bound on the range of the PDF.
        upper: Upper bound on the range of the PDF.

    Return:
        normalized_pdf: Function for computing the normalized PDF.
    """
    norm = get_normalization_1d_pdf(pdf, lower, upper)

    def normalized_pdf(*args) -> NDArray:
        """Get the function values for the normalized PDF."""
        return norm(*args) * pdf(*args)

    return normalized_pdf


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
        llh_pdf_mesh: The maximum log-likelihood.
        max_llh_par1: Maximum likelihood estimate of par1.
        max_llh_par2: Maximum likelihood estimate of par2.
    """
    par1_values = np.linspace(*par1_range, N)
    par2_values = np.linspace(*par2_range, N)
    par1_mesh, par2_mesh = np.meshgrid(par1_values, par2_values)
    pdf_mesh = pdf(
        data.reshape(1, -1),
        par1_mesh.reshape(-1, 1),
        par2_mesh.reshape(-1, 1),
    )
    llh_pdf_mesh = np.sum(np.log(pdf_mesh), axis=1).reshape(N, N)
    max_llh_idx = np.argmax(llh_pdf_mesh.ravel())
    max_llh_par1 = par1_mesh.ravel()[max_llh_idx]
    max_llh_par2 = par2_mesh.ravel()[max_llh_idx]
    return par1_mesh, par2_mesh, llh_pdf_mesh, max_llh_par1, max_llh_par2
