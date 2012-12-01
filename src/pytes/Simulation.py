import numpy as np
import pyfits
from pytes import Analysis, Pulse

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

def simulate(N, sigma, noise=3e-6, sps=1e6, t=2e-3, Emax=7000, atom="Mn"):
    """
    Simulate
    """
    
    # Generate distribution
    pdf = lambda E: Analysis.line_model(E, sigma, line=atom+"Ka")
    e = random(pdf, int(N*0.9), 0, Emax)
    
    pdf = lambda E: Analysis.line_model(E, sigma, line=atom+"Kb")
    e = np.concatenate((e, random(pdf, int(N*0.1), 0, Emax)))
    
    # Convert energy to PHA
    pha = e / Emax
    
    # Generate pulses and noises
    points = int(sps*t)
    pulse = np.array([ Pulse.dummy(p, points=points, t=t, duty=0.1) +
                        Pulse.white(var=noise**2, points=points, t=t) for p in pha ])
    
    noise = np.array([ Pulse.white(var=noise**2, points=points, t=t) for p in pha ])
    
    return pulse, noise