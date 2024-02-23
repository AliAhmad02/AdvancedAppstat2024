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
