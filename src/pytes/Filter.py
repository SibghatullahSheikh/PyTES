#
# Optimal Filter
#
# K. Sakai (sakai@astro.isas.jaxa.jp)

import numpy as np

def median_filter(arr, sigma):
    """
    Noise filter using Median and Median Absolute Deviation for 1-dimentional array
    """
                
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
                
    # Tiny cheeting for mad = 0 case
    if mad == 0:
        absl = np.abs(arr - med)
        if len(absl[absl > 0]) > 0:
            mad = (absl[absl > 0])[0]
        else:
            mad = np.std(arr) / 1.4826
                
    return (arr >= med - mad*1.4826*sigma) & (arr <= med + mad*1.4826*sigma)

def reduction(data, sigma=3, **kwargs):
    """
    Do data reduction with sum, max and min for pulse/noise using median filter (or manual min/max)
    
    Parameters (and their default values):
        data:   array of pulse/noise data (NxM or N array-like)
        sigma:  sigmas allowed for median filter
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (mask)
        mask:   boolean array for indexing filtered data
    """
    
    data = np.array(data)
    
    if kwargs.has_key("min"):
        min_mask = (data.min(axis=1) > kwargs["min"][0]) & (data.min(axis=1) < kwargs["min"][1])
    else:
        min_mask = median_filter(data.min(axis=1), sigma)

    if kwargs.has_key("max"):
        max_mask = (data.max(axis=1) > kwargs["max"][0]) & (data.max(axis=1) < kwargs["max"][1])
    else:
        max_mask = median_filter(data.max(axis=1), sigma)

    if kwargs.has_key("sum"):
        sum_mask = (data.sum(axis=1) > kwargs["sum"][0]) & (data.sum(axis=1) < kwargs["sum"][1])
    else:
        sum_mask = median_filter(data.sum(axis=1), sigma)

    return min_mask & max_mask & sum_mask

def average_pulse(pulse, sigma=3, max_shift=None, **kwargs):
    """
    Generate an averaged pulse
    
    Parameters (and their default values):
        pulse:      array of pulse data (NxM or N array-like, obviously the latter makes no sense though)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation (Default: None = length / 2)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (averaged_pulse)
        averaged_pulse:     averaged pulse
    """
    
    # if given data is not numpy array, convert them
    if type(pulse) is np.ndarray:
        pulse = np.array(pulse)
    else:
        # Copy it if it is already numpy array
        pulse = pulse.copy()
    
    # Calculate averaged pulse
    if pulse.ndim == 2:
        # Data reduction
        if sigma is not None:
            pulse = pulse[reduction(pulse, sigma, **kwargs)]

        # Align pulses to the first pulse
        if len(pulse) > 1:
            for i in range(1, len(pulse)):
                s = cross_correlate(pulse[0], pulse[i], max_shift=max_shift)[1]
                pulse[i] = np.roll(pulse[i], s)

        avg_pulse = np.average(pulse, axis=0)
    
    elif pulse.ndim == 1:
        # Only one pulse data. No need to average
        avg_pulse = pulse
    
    else:
        raise ValueError("object too deep for desired array")
    
    return avg_pulse

def average_noise(noise, sigma=3, **kwargs):
    """
    Calculate averaged noise power
    
    Parameters (and their default values):
        noise:      array of pulse data (NxM or N array-like, obviously the latter makes no sense though)
        sigma:      sigmas allowed for median filter, or None to disable filtering (Default: 3)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (averaged_pulse)
        power_noise:    calculated averaged noise power
    """

    # Convert to numpy array
    noise = np.array(noise)

    if sigma is not None:
        noise = noise[reduction(noise, sigma, **kwargs)]

    return np.average(np.abs(np.fft.rfft(noise))**2, axis=0)

def generate_template(pulse, noise, cutoff=None, **kwargs):
    """
    Generate a template of optimal filter

    Parameters (and their default values):
        pulse:  array of pulse data, will be averaged when used
        noise:  array of noise data, will be averaged in frequency domain when used
        cutoff: cut-off bin number for pulse spectrum (Default: None = no cut-off)
    
    Valid keywords:
        min:    tuple of (min, max) for min
        max:    tuple of (min, max) for max
        sum:    tuple of (min, max) for sum
    
    Return (template)
        template:   generated template
    """
    
    # Sanity check
    assert pulse.ndim == 2, "dimension of pulse must be 2"    
    assert noise.ndim == 2, "dimension of noise must be 2"
    assert pulse.shape[1] == noise.shape[1], "data length needs to be matched"
    
    # Average pulse
    avg_pulse = average_pulse(pulse, **kwargs)

    # Real-DFT
    fourier = np.fft.rfft(avg_pulse)
    
    # Apply cut-off filter (hanning filter) if needed
    if 0 < cutoff < len(avg_pulse)/2 + 1:
        fourier *= np.hstack((0.5 * (1.0 + np.cos(np.pi * np.arange(cutoff) / np.float(cutoff))),
                                np.zeros(len(fourier) - cutoff)))    
    
    # Calculate averaged noise power    
    pow_noise = average_noise(noise, **kwargs)
    
    # Calculate S/N ratio
    sn = np.abs(fourier)/np.sqrt(pow_noise)
    
    # Generate template (inverse Real-DFT)
    fourier[0] = 0  # Eliminate DC
    template = np.fft.irfft(fourier / pow_noise)
    
    # Normalize template
    norm = (avg_pulse.max() - avg_pulse.min()) / ((template * avg_pulse).sum() / len(avg_pulse))
    
    return template * norm, sn

def cross_correlate(data1, data2, max_shift=None):
    """
    Calculate a cross correlation for a given set of data.
    
    Parameters (and their default values):
        data1:      pulse/noise data (array-like)
        data2:      pulse/noise data (array-like)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
    
    Return (max_cor, shift)
        max_cor:    calculated max cross correlation
        shift:      required shift to maximize cross correlation
    """

    # Sanity check
    if len(data1) != len(data2):
        raise ValueError("data length does not match")

    # if given data set is not numpy array, convert them
    if type(data1) is not np.ndarray:
        data1 = np.array(data1)
    
    if type(data2) is not np.ndarray:
        data2 = np.array(data2)
    
    # Calculate cross correlation
    if max_shift == 0:
        return np.correlate(data1, data2, 'valid')[0] / len(data1), 0
    
    # Needs shift
    if max_shift is None:
        max_shift = len(data1) / 2
    else:
        # max_shift should be less than half data length
        max_shift = min(max_shift, len(data1) / 2)
    
    # Calculate cross correlation
    cor = np.correlate(data1, np.concatenate((data2[-max_shift:], data2, data2[:max_shift])), 'valid')
    ind = cor.argmax()
    
    return cor[ind] / len(data1), ind - max_shift

def optimal_filter(pulse, template, max_shift=None):
    """
    Perform an optimal filtering for pulse using template
    
    Parameters (and their default values):
        pulse:      pulses (NxM array-like)
        template:   template (array-like)
        max_shift:  maximum allowed shifts to calculate maximum cross correlation
                    (Default: None = length / 2)
    
    Return (pha)
        pha:        pha array
    """
    
    return np.apply_along_axis(lambda p: cross_correlate(template, p, max_shift=max_shift)[0], 1, pulse)

def offset(pulse, bins=None):
    """
    Calculate an offset (DC level) of pulses
    
    Parameters (and their default values):
        pulse:  pulses (N or NxM array-like)
        bins:   tuple of (start, end) for bins used for calculating an offset
                (Default: None = automatic determination)
    
    Return (offset)
        offset: calculated offset level
    """
    
    pulse = np.array(pulse)
    
    if bins is None:
        # Determine end bin for offset calculation (start is always 0)
        if pulse.ndim == 1:
            return pulse[:np.correlate(pulse, [1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1]).argmax() + 9].mean()
        else:
            return np.array([ p[:np.correlate(p, [1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,1,1]).argmax() + 9].mean() for p in pulse ])
    else:
        if pulse.ndim == 1:
            return pulse[bins[0]:bins[1]+1].mean()
        else:
            return pulse[:,bins[0]:bins[1]+1].mean(axis=1)