import numpy as np
from scipy.stats import norm

def dummy(pha=1.0, tr=2e-6, tf=100e-6, points=1024, t=2e-3, duty=0.5, talign=0.0):
    """
    Generate Dummy Pulse
    """
    
    ts = np.linspace(-t*duty, t*(1-duty), points) + t/points*talign
    
    def M(t):
        return 0.0 if t < 0 else (tr+tf)/tf**2 * (1-np.exp(-t/tr)) * np.exp(-t/tf)
    Mvec = np.vectorize(M)
    
    return pha*tf*Mvec(ts)

def white(var=3e-6**2, mean=0.0, t=2e-3, points=1024):
    """
    Generate Gaussian White Noise
    """
    
    return (mean + np.sqrt(var)*norm.rvs(size=points)) / np.sqrt(t/(points[0] if type(points) in (list, tuple, np.array) else points))