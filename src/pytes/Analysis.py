import numpy as np
import warnings
from numpy.linalg import lstsq
from scipy.special import wofz
from scipy.optimize import curve_fit, minimize
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

def line_model(E, dE=0, width=0, line="MnKa", shift=False, full=False):
    """
    Line model
    
    Parameters (and their default values):
        E:      energy in eV (array-like)
        dE:     shift from center energy in eV (Default: 0 eV)
        width:  FWHM of gaussian profile in eV (Default: 0 eV)
        line:   line (Default: MnKa)
        shift:  treat dE as shift if True instead of scaling (Default: False)
        full:   switch for return value (Default: False)
    
    Return (i) when full = False or (i, i1, i2, ...) when full = True
        i:      total intensity
        i#:     component intensities
    
    Note:
        If shift is False, adjusted center energies ec_i of fine structures
        will be
        
            ec_i = Ec_i * (1 + dE/Ec)
        
        where Ec_i is the theoretical (experimental) center energy of fine
        structures and Ec is the center energy of the overall profile, which
        is the weighted sum of each component profiles.
        
        If shift is True, ec_i will simply be
        
            ec_i = Ec_i + dE.
    """
    
    # Sanity check
    if not FS.has_key(line):
        raise ValueError("No data for %s" % line)
    
    # Center energy
    Ec = np.exp(np.log(np.asarray(FS[line])[:,(0,2)]).sum(axis=1)).sum()
    
    if shift:
        if width == 0:
            model = np.array([ p[2] * lorentzian(E, p[0]+dE, p[1]) for p in FS[line] ])
        else:
            model = np.array([ p[2] * voigt(E, p[0]+dE, p[1], width) for p in FS[line] ])
    else:
        if width == 0:
            model = np.array([ p[2] * lorentzian(E, p[0]*(1+dE/Ec), p[1]) for p in FS[line] ])
        else:
            model = np.array([ p[2] * voigt(E, p[0]*(1+dE/Ec), p[1], width) for p in FS[line] ])

    if full:
        return np.vstack((model.sum(axis=0)[np.newaxis], model))
    else:
        return model.sum(axis=0)

def group_bin(n, bins, min=100):
    """
    Group PHA bins to have at least given number of minimum counts
    
    Parameters (and their default values):
        n:      counts
        bins:   bin edges
        min:    minimum counts to group (Default: 100)
    
    Return (grouped_n, grouped_bins)
        grouped_n:      grouped counts
        grouped_bins:   grouped bin edges
    """
    
    grp_n = []
    grp_bins = [bins[0]]

    n_sum = 0

    for p in zip(n, bins[1:]):
        n_sum += p[0]
        
        if n_sum >= min:
            grp_n.append(n_sum)
            grp_bins.append(p[1])
            n_sum = 0
    
    return np.asarray(grp_n), np.asarray(grp_bins)

def histogram(pha, binsize=1.0):
    """
    Create histogram
    
    Parameter:
        pha:        pha data (array-like)
        binsize:    size of bin in eV (Default: 1.0 eV)
    
    Return (n, bins)
        n:      photon count
        bins:   bin edge array
    
    Note:
        - bin size is 1eV/bin.
    """
    
    # Create histogram
    bins = np.arange(np.floor(pha.min()), np.ceil(pha.max())+binsize, binsize) 
    n, bins = np.histogram(pha, bins=bins)
    
    return n, bins

def _fit_cs(pha, binsize=1, min=20, line="MnKa", shift=False):
    """
    Chi-squared Fitting of line spectrum by Voigt profiles
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        binsize:    size of energy bin in eV for histogram (Default: 1 eV)
        min:        minimum counts to group bins (Default: 20 bins)
        line:       line to fit (Default: MnKa)
        shift:      treat dE as shift if True instead of scaling (Default: False)
    
    Return (dE, width), (dE_error, width_error), (chi_squared, dof)
        dE:             shift from line center
        width:          fitted gaussian width (FWHM)
        dE_error:       dE error (1-sigma)
        width_error:    width error (1-sigma)
        chi_squared:    chi^2
        dof:            degrees of freedom
    """
    
    # Sanity check
    if not FS.has_key(line):
        raise ValueError("No data for %s" % line)
    
    # Create histogram
    n, bins = histogram(pha, binsize=binsize)
    
    # Group bins
    gn, gbins = group_bin(n, bins, min)
    ngn = gn/(np.diff(gbins))   # normalized counts in counts/eV
    ngn_sigma = np.sqrt(gn)/(np.diff(gbins))
    
    bincenters = (gbins[1:]+gbins[:-1])/2
    
    # Fit
    def model(E, dE, A, width):
        return A * line_model(E, dE, width, line, shift)
    
    popt, pcov = curve_fit(model, bincenters, ngn,
                        p0=(0, max(ngn), ngn.sum()/max(ngn)), sigma=ngn_sigma)

    if len(pcov) == 1:
        raise Exception("Fitting failed for %s" % line)
    
    dE, A, width = popt
    dE_e, A_e, width_e = np.sqrt(np.diag(pcov))
    
    # Calculate chi_squared
    chi_squared = (((ngn - model(bincenters, dE, A, width))/ngn_sigma)**2).sum()
    dof = len(bincenters) - 3
    
    return (dE, width), (dE_e, width_e), (chi_squared, dof)

def _fit_ls(pha, binsize=1, min=20, line="MnKa", shift=False):
    """
    Least-squared Fitting of line spectrum by Voigt profiles
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        binsize:    size of energy bin in eV for histogram (Default: 1 eV)
        min:        minimum counts to group bins (Default: 20 bins)
        line:       line to fit (Default: MnKa)
        shift:      treat dE as shift if True instead of scaling (Default: False)
    
    Return (dE, width), (dE_error, width_error), (None)
        dE:             shift from line center
        width:          fitted gaussian width (FWHM)
        dE_error:       dE error (1-sigma)
        width_error:    width error (1-sigma)
    """
    
    # Sanity check
    if not FS.has_key(line):
        raise ValueError("No data for %s" % line)
    
    # Create histogram
    n, bins = histogram(pha, binsize=binsize)
    
    # Group bins
    gn, gbins = group_bin(n, bins, min)
    ngn = gn/(np.diff(gbins))   # normalized counts in counts/eV
    ngn_sigma = np.sqrt(gn)/(np.diff(gbins))
    
    bincenters = (gbins[1:]+gbins[:-1])/2
    
    # Fit
    def model(E, dE, A, width):
        return A * line_model(E, dE, width, line, shift)
    
    popt, pcov = curve_fit(model, bincenters, ngn,
                        p0=(0, max(ngn), ngn.sum()/max(ngn)))

    if len(pcov) == 1:
        raise Exception("Fitting failed for %s" % line)
    
    dE, A, width = popt
    dE_e, A_e, width_e = np.sqrt(np.diag(pcov))
    
    return (dE, width), (dE_e, width_e), (None)

def _fit_mle(pha, line="MnKa", shift=False):
    """
    Maximum-likelihood estimation of line spectrum by Voigt profiles
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        line:       line to fit (Default: MnKa)
        shift:      treat dE as shift if True instead of scaling (Default: False)
    
    Return (dE, width), (None, None), (None)
        dE:             shift from line center
        width:          fitted gaussian width (FWHM)
    """
    
    def lf((dE, width)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return -np.log(line_model(pha, dE, width, line, shift)).sum()
    
    res = minimize(lf, x0=(0, 1), method="BFGS", options={"gtol": 1e-2})
    
    if not res.success:
        raise RuntimeError('MLE failed')
    
    return res.x, np.sqrt(np.diag(np.linalg.inv(res.hess))), (None)

def fit(pha, binsize=1, min=20, line="MnKa", shift=False, method='mle'):
    """
    Fit line spectrum by Voigt profiles
    
    Parameters (and their default values):
        pha:        pha data (array-like)
        binsize:    size of energy bin in eV for histogram (Default: 1 eV)
        min:        minimum counts to group bins (Default: 20 bins)
        line:       line to fit (Default: MnKa)
        shift:      treat dE as shift if True instead of scaling (Default: False)
        method:     fitting method from mle (maximum-likelihood estimate),
                    cs (chi-squared) or ls (least-squared) (Default: mle)
    
    Return (dE, width), (dE_error, width_error), (chi_squared, dof)
        dE:             shift from line center
        width:          fitted gaussian width (FWHM)
        dE_error:       dE error (1-sigma) (only in cs and ls)
        width_error:    width error (1-sigma) (only in cs and ls)
        chi_squared:    chi^2 (only in cs)
        dof:            degrees of freedom (only in cs)
    """
    
    if method.lower() == "mle":
        return _fit_mle(pha, line=line, shift=shift)
    elif method.lower() == "cs":
        return _fit_cs(pha, binsize=binsize, min=min, line=line, shift=shift)
    elif method.lower() == "ls":
        return _fit_ls(pha, binsize=binsize, min=min, line=line, shift=shift)
    else:
        raise ValueError('Unknown method: %s' % method)

def normalization(n, bins, dE, width, line="MnKa", shift=False):
    """
    Estimate model normalization
    
    Parameters (and their default values):
        n:      counts
        bins:   bin edges
        dE:     shift from line center
        width:  fitted gaussian width (FWHM)
        line:   line to fit (Default: MnKa)
        shift:  treat dE as shift if True instead of scaling (Default: False)
    
    Return (norm)
        norm:   normalization
    """
    
    # Model
    def model(E, A):
        return A * line_model(E, dE, width, line, shift)
    
    bincenters = (bins[1:]+bins[:-1])/2
    
    popt, pcov = curve_fit(model, bincenters, n, p0=(max(n)))
    
    return popt[0]
    
    