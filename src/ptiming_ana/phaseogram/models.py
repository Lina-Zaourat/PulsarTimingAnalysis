import numpy as np
import numba as nb

__all__ = [
    "get_model_list",
    "gaussian",
    "double_gaussian",
    "double_gaussian_heaviside", #(LBZ)
    "triple_gaussian",
    "assymetric_gaussian_pdf",
    "assymetric_double_gaussian",
    "lorentzian",
    "double_lorentz",
    "lorentz_pdf",
    "step_function", #(LBZ)
    "dgaussian_step", #(LBZ)
]


def get_model_list():
    return [
        "gaussian",
        "dgaussian",
        "dgaussian_heaviside", #(LBZ)
        "double_lorentz",
        "asym_dgaussian",
        "tgaussian",
        "lorentzian",
        "step", #(LBZ)
        "dgaussian_step", #(LBZ)
    ]


kwd = {"parallel": False, "fastmath": True}


@nb.njit(**kwd)
def gaussian(x, mu, sigma, A, B):
    return A + B / np.sqrt(2 * np.pi) / sigma * np.exp(
        -((x - mu) ** 2) / 2.0 / sigma**2
    )


@nb.njit(**kwd)
def double_gaussian(x, mu, sigma, mu_2, sigma_2, A, B, C):
    return (
        A
        + B / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mu) ** 2) / 2.0 / sigma**2)
        + C
        / (2 * np.pi) ** (1 / 2)
        / sigma_2
        * np.exp(-((x - mu_2) ** 2) / 2.0 / sigma_2**2)
    )


def double_gaussian_heaviside(x, mu, sigma, mu_2, sigma_2, mu_3, A, B, C, D): #(LBZ)
    return ( #(LBZ)
        A #(LBZ)
        + B / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mu) ** 2) / 2.0 / sigma**2) #(LBZ)
        + C #(LBZ)
        / np.sqrt(2 * np.pi) #(LBZ)
        / sigma_2 #(LBZ)
        * np.exp(-((x - mu_2) ** 2) / 2.0 / sigma_2**2) #(LBZ)
        + D * np.heaviside(x - mu_3,1 ) #(LBZ)
    )


@nb.njit(**kwd)
def triple_gaussian(x, Bkg, mu, sigma, mu_2, sigma_2, mu_3, sigma_3, A, B, C):
    return (
        Bkg
        + A / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mu) ** 2) / 2.0 / sigma**2)
        + B
        / np.sqrt(2 * np.pi)
        / sigma_2
        * np.exp(-((x - mu_2) ** 2) / 2.0 / sigma_2**2)
        + C
        / np.sqrt(2 * np.pi)
        / sigma_3
        * np.exp(-((x - mu_3) ** 2) / 2.0 / sigma_3**2)
    )


@nb.njit(**kwd)
def assymetric_gaussian_pdf(x, mu, sigma1, sigma2):
    if x <= mu:
        return (
            2
            / np.sqrt(2 * np.pi)
            / (abs(sigma1) + abs(sigma2))
            * np.exp(-((x - mu) ** 2) / 2.0 / sigma1**2)
        )
    else:
        return (
            2
            / np.sqrt(2 * np.pi)
            / (abs(sigma1) + abs(sigma2))
            * np.exp(-((x - mu) ** 2) / 2.0 / sigma2**2)
        )


@nb.njit(**kwd)
def assymetric_double_gaussian(
    x, mu, sigma1, sigma2, mu_2, sigma1_2, sigma2_2, A, B, C
):
    return (
        A
        + B * assymetric_gaussian_pdf(x, mu, sigma1, sigma2)
        + C * assymetric_gaussian_pdf(x, mu_2, sigma1_2, sigma2_2)
    )


@nb.njit(**kwd)
def lorentz_pdf(x, mu, gamma):
    return 1 / (np.pi * gamma) * (gamma**2) / ((x - mu) ** 2 + gamma**2)


def lorentzian(x, mu, gamma, A, B):
    return A + B / (np.pi * gamma) * (gamma**2) / ((x - mu) ** 2 + gamma**2)


@nb.njit(**kwd)
def double_lorentz(x, mu_1, gamma_1, mu_2, gamma_2, A, B, C):
    # lorentz_pdf_vec=np.vectorize(lorentz_pdf)
    return A + B * lorentz_pdf(x, mu_1, gamma_1) + C * lorentz_pdf(x, mu_2, gamma_2)


def step_function(x, phi1, phi2, A, B): #(LBZ)
    """
    Step/Rectangle function for modeling a phase interval (e.g., P3 peak).
    
    Mathematical model:
        f(x; φ₁, φ₂, A, B) = A + B  if φ₁ ≤ x ≤ φ₂
                            = A      otherwise
    
    Equivalent form using Heaviside function:
        f(x) = A + B·H(x - φ₁)·H(φ₂ - x)
    
    Parameters:
        x (array): phase values
        phi1 (float): start of the phase interval
        phi2 (float): end of the phase interval  
        A (float): background/baseline level
        B (float): height of the step/rectangle
    
    Returns:
        array: f(x) = A + B in [φ₁, φ₂], A elsewhere
    """
    return A + B * np.heaviside(x - phi1, 1.0) * np.heaviside(phi2 - x, 1.0) #(LBZ)


def dgaussian_step(x, mu, sigma, mu_2, sigma_2, phi1, phi2, A, B, C, D): #(LBZ)
    """
    Double Gaussian + Step function model.
    Combines P1 and P2 as Gaussians with P3 as a rectangular step.
    
    Mathematical model:
        f(x; μ, σ, μ₂, σ₂, φ₁, φ₂, A, B, C, D) =
            A + B/√(2π)/σ * exp(-((x-μ)²)/(2σ²))
              + C/√(2π)/σ₂ * exp(-((x-μ₂)²)/(2σ₂²))
              + D·H(x - φ₁)·H(φ₂ - x)
    
    Parameters:
        x (array): phase values
        mu (float): mean of P1 (Gaussian)
        sigma (float): std dev of P1
        mu_2 (float): mean of P2 (Gaussian)
        sigma_2 (float): std dev of P2
        phi1 (float): start of P3 (step)
        phi2 (float): end of P3 (step)
        A (float): baseline/background
        B (float): amplitude of P1
        C (float): amplitude of P2
        D (float): amplitude of P3 (step height)
    
    Returns:
        array: Combined model with 3 peaks
    """
    return ( #(LBZ)
        A #(LBZ)
        + B / np.sqrt(2 * np.pi) / sigma * np.exp(-((x - mu) ** 2) / 2.0 / sigma**2) #(LBZ) P1
        + C / np.sqrt(2 * np.pi) / sigma_2 * np.exp(-((x - mu_2) ** 2) / 2.0 / sigma_2**2) #(LBZ) P2
        + D * np.heaviside(x - phi1, 1.0) * np.heaviside(phi2 - x, 1.0) #(LBZ) P3 (step)
    )
