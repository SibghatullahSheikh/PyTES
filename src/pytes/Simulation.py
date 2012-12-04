import numpy as np
import pyfits
from pytes import Analysis, Pulse, Constants

def random(pdf, N, min, max):
    """
    Generate random values in given distribution using the rejection method
    
    Parameters:
        pdf:    distribution function
        N:      desired number of random values
        min:    minimum of random values
        max:    maximum of random values
    
    Return (values)
        values: generated random values
    """
    
    valid = np.array([])
    
    while len(valid) < N:
        r = np.random.uniform(min, max, N)
        p = np.random.uniform(0, 1, N)
        
        valid = np.concatenate((valid, r[p < pdf(r)]))
    
    return valid[:N]

def simulate(N, width, noise=3e-6, sps=1e6, t=2e-3, Emax=10e3, atom="Mn"):
    """
    Generate pulse (Ka and Kb) and noise
    
    Parameters (and their default values):
        N:      desired number of pulses/noises
        width:  width (FWHM) of gaussian (voigt) profile
        noise:  white noise level in V/srHz (Default: 3uV/srHz)
        sps:    sampling per second (Default: 1Msps)
        t:      sampling time (Default: 2ms)
        Emax:   max energy in eV (Default: 10keV)
        atom:   atom to simulate (Default: Mn)
    
    Return (pulse, noise):
        pulse:  simulated pulse data (NxM array-like)
        noise:  simulated noise data (NxM array-like)
    
    Note:
        - pha 1.0 = Emax
    """
    
    # Simulate Ka and Kb lines
    pdf = lambda E: Analysis.line_model(E, width, line=atom+"Ka")
    Ec = np.array(Constants.FS[atom+"Ka"])[:,0]
    _Emin = np.min(Ec) - width*500
    _Emax = np.max(Ec) + width*500
    e = random(pdf, int(N*0.9), _Emin, _Emax)
    
    pdf = lambda E: Analysis.line_model(E, width, line=atom+"Kb")
    Ec = np.array(Constants.FS[atom+"Kb"])[:,0]
    _Emin = np.min(Ec) - width*500
    _Emax = np.max(Ec) + width*500
    e = np.concatenate((e, random(pdf, int(N*0.1), _Emin, _Emax)))
    
    # Convert energy to PHA
    pha = e / Emax
    
    # Generate pulses and noises
    points = (N, int(sps*t))
    pulse = Pulse.dummy(pha, points=points, t=t, duty=0.1,
                        talign=np.random.uniform(size=N)-0.5) + \
            Pulse.white(var=noise**2, points=points, t=t)
    
    noise = Pulse.white(var=noise**2, points=points, t=t)
    
    return pulse, noise