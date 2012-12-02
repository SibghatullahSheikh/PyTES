import numpy as np
from scipy.stats import norm

def dummy(pha=1.0, tr=2e-6, tf=100e-6, points=1024, t=2e-3, duty=0.5, talign=0.0):
    """
    Generate Dummy Pulse
    """
    
    points = np.asarray(points)
    
    # Data counts
    c = 1 if points.ndim == 0 else np.ones(points[0])[np.newaxis].T
    
    # Data length
    l = points if points.ndim == 0 else points[-1]
    
    # Time steps
    ts = np.linspace(-t*duty, t*(1-duty), l)*c + t/l*np.asarray(talign)[np.newaxis].T
    
    # Pulse model function
    def M(t):
        return 0.0 if t < 0 else (tr+tf)/tf**2 * (1-np.exp(-t/tr)) * np.exp(-t/tf)
    
    # Vectorize pulse model function
    Mvec = np.vectorize(M)
    
    return np.asarray(pha)[np.newaxis].T*tf*Mvec(ts)

def white(var=3e-6**2, mean=0.0, t=2e-3, points=1024):
    """
    Generate Gaussian White Noise
    """
    
    points = np.asarray(points)
    
    # Data length
    l = points if points.ndim == 0 else points[-1]
    
    # Time resolution of Nyquist frequency
    dt = t / l * 2

    return (mean + np.sqrt(var)*norm.rvs(size=points)) / np.sqrt(dt)