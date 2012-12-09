import numpy as np
from numpy.linalg import lstsq
from scipy.special import wofz
from scipy.optimize import curve_fit
from Filter import median_filter
from Constants import *

def baseline(sn, E=5.9e3):
    """
    Calculate a baseline resolution dE(FWHM) for the given energy.
    
    Parameter:
        sn:     S/N ratio (array-like)
        E:      energy to calculate dE
    """
    
    return 2*np.sqrt(2*np.log(2))*E/np.sqrt((sn**2).sum()*2)

def ka(pha, sigma=1):
    """
    Return a k-alpha data set.
    
    Parameters (and their default values):
        pha:    pha (and optional companion) data (array-like)
        sigma:  sigmas allowed for median filter (Default: 1)

    Return (data)
        data:   k-alpha data set
    """
    
    pha = np.asarray(pha)
    
    if pha.ndim > 1:
        mask = median_filter(pha[:,0], sigma=sigma)
    else:
        mask = median_filter(pha, sigma=sigma)
    
    return pha[mask]

def kb(pha, sigma=1):
    """
    Find a k-beta data set.
    
    Parameters (and their default values):
        pha:    pha (and optional companion) data (array-like)
        sigma:  sigmas allowed for median filter (Default: 1)

    Return (data)
        data:   k-beta data set
    """
    
    pha = np.asarray(pha)
    
    if pha.ndim > 1:
        _pha = pha[:,0]
    else:
        _pha = pha
    
    ka_mean = ka(_pha).mean()
    ka_mask = median_filter(_pha, sigma=sigma)
    
    kb_mask = median_filter(_pha[(ka_mask == False) & (_pha>ka_mean)], sigma=sigma)
    kb_pha  = pha[(ka_mask == False) & (_pha>ka_mean)]
    
    return kb_pha[kb_mask]

def offset_correction(pha, offset, sigma=1):
    """
    Perform an offset (DC level) correction for PHA
    
    Parameters (and their default values):
        pha:    pha data (array-like)
        offset: offset data (array-like)
        sigma:  sigmas allowed for median filter (Default: 1)
    
    Return (pha):
        pha:    corrected pha data
    """
    
    # Sanity check
    if len(pha) != len(offset):
        raise ValueError("data length of pha and offset does not match")
    
    # Zip pha and offset
    data = np.vstack((pha, offset)).T
    
    # Correction using K-alpha
    ka_pha, ka_offset = ka(data, sigma=sigma).T
    ka_a, ka_b = np.polyfit(ka_offset, ka_pha, 1)
    corrected_pha = pha/(ka_a*offset+ka_b)*ka_b
    
    return corrected_pha

def linearity_correction(pha, atom="Mn", sigma=1):
    """
    Perform a linearity correction for PHA
    
    Parameters (and their default values):
        pha:    pha data (array-like)
        atom:   atom to use for correction (Default: Mn)
        sigma:  sigmas allowed for median filter (Default: 1)
    
    Return (pha):
        pha:    corrected pha data
    """
    
    # Mn Ka/Kb Energy
    if not LE.has_key(atom):
        raise ValueError("No data for %s" % atom)
    
    Mn = np.asarray(LE[atom])
    
    # MnKa/MnKb PHA Center
    pha_center = np.array([ ka(pha, sigma=sigma).mean(), kb(pha, sigma=sigma).mean() ])
    
    # Fitting
    p = lstsq(np.vstack((Mn**2, Mn)).T, pha_center)[0]
    
    # Correction
    corrected_pha = (-p[1] + np.sqrt(p[1]**2+4*p[0]*pha)) / (2*p[0])
    
    return corrected_pha

def voigt(E, Ec, nw, gw):
    """
    Voigt profile
     
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural (lorentzian) width (FWHM)
        gw:     gaussian width (FWHM)
    
    Return (voigt)
        voigt:  Voigt profile
    """
     
    z = (E - Ec + 1j*fwhm2gamma(nw)) / (fwhm2sigma(gw)*np.sqrt(2))

    return wofz(z).real / (fwhm2sigma(gw)*np.sqrt(2*np.pi))

def lorentzian(E, Ec, nw):
    """
    Lorentzian profile
    
    Parameters:
        E:      energy
        Ec:     center energy
        nw:     natural width (FWHM)
    
    Return (lorentz)
        lorentz:  Lorentzian profile
    """

    gamma = fwhm2gamma(nw)

    return 1.0/np.pi * (gamma / ((E-Ec)**2 + gamma**2))

def sigma2fwhm(sigma):
    """
    Convert sigma to width (FWHM)
    
    Parameter:
        sigma:  sigma of gaussian / voigt profile
    
    Return (fwhm)
        fwhm:   width
    """
    
    return 2*sigma*np.sqrt(2*np.log(2))

def fwhm2sigma(fwhm):
    """
    Convert width (FWHM) to sigma
    
    Parameter:
        fwhm:   width
    
    Return (sigma)
        sigma:  sigma of gaussian / voigt profile
    """
    
    return fwhm/(2*np.sqrt(2*np.log(2)))

def gamma2fwhm(gamma):
    """
    Convert gamma to width (FWHM)
    
    Parameter:
        gamma:  gamma of lorentzian / voigt profile
    
    Return (fwhm)
        fwhm:   width
    """
    
    return gamma*2.0

def fwhm2gamma(fwhm):
    """
    Convert width (FWHM) to gamma
    
    Parameter:
        fwhm:   width
    
    Return (sigma)
        gamma:  gamma of lorentzian / voigt profile
    """
    
    return fwhm/2.0

def line_model(E, dE=0, width=0, line="MnKa", full=False):
    """
    Line model
    
    Parameters (and their default values):
        E:      energy in eV (array-like)
        dE:     shift from center energy in eV (Default: 0 eV)
        width:  FWHM of gaussian profile in eV (Default: 0 eV)
        line:   line (Default: MnKa)
        full:   switch for return value (Default: False)
    
    Return (i) when full = False or (i, i1, i2, ...) when full = True
        i:      total intensity
        i#:     component intensities
    """
    
    # Sanity check
    if not FS.has_key(line):
        raise ValueError("No data for %s" % line)
    
    # Center energy
    Ec = np.exp(np.log(np.asarray(FS[line])[:,(0,2)]).sum(axis=1)).sum()
    
    if width == 0:
        model = np.array([ p[2] * lorentzian(E, p[0]*(1+dE/Ec), p[1]) for p in FS[line] ])
    else:
        model = np.array([ p[2] * voigt(E, p[0]*(1+dE/Ec), p[1], width) for p in FS[line] ])
    
    if full:
        return np.vstack((model.sum(axis=0)[np.newaxis], model))
    else:
        return model.sum(axis=0)

def fit(pha, bins=40, line="MnKa"):
    """
    Fit line spectrum by Voigt profiles
    
    Parameters (and their default values):
        pha:    pha data (array-like)
        bins:   histogram bins (Default: 40)
        line:   line to fit (Default: MnKa)
    
    Return (dE, dE_error, A, A_error, sigma, sigma_error)
        dE:             shift from line center
        dE_error:       dE error (1-sigma)
        A:              fitted amplitude
        A_error:        amplitude error (1-sigma)
        width:          fitted gaussian width (FWHM)
        width_error:    width error (1-sigma)
    """
    
    # Sanity check
    if not FS.has_key(line):
        raise ValueError("No data for %s" % line)
    
    # Create histogram
    n, bins = np.histogram(pha, bins=bins, density=True)
    bincenters = (bins[1:]+bins[:-1])/2
    
    # Fit
    def model(E, dE, A, width):
        return A * line_model(E, dE, width, line)
    
    popt, pcov = curve_fit(model, bincenters, n, p0=(0, max(n), 1/max(n)))
    
    return popt[0], np.sqrt(pcov[0][0]), popt[1], np.sqrt(pcov[1][1]), popt[2], np.sqrt(pcov[2][2])