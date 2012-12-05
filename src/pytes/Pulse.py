import numpy as np
from scipy.stats import norm

def pulse(pha=1.0, tr=2e-6, tf=100e-6, points=1024, t=2e-3, duty=0.5, talign=0.0):
    """
    Generate Dummy Pulse
    
    Parameters (and their default values):
        pha:    pulse height (Default: 1.0)
        tr:     rise time constant (Default: 2 us)
        tf:     fall time constant (Default: 100 us)
        points: data length (scalar or 2-dim tuple/list) (Default: 1024)
        t:      sample time (Default: 2 ms)
        duty:   pulse raise time ratio to t (Default: 0.5)
                should take from 0.0 to 1.0
        talign: time offset ratio to dt (=t/points) (Default: 0)
                should take from -0.5 to 0.5
    
    Return (pulse)
        pulse:  pulse data (array-like)
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
        return 0.0 if t < 0 else (1-np.exp(-t/tr)) * np.exp(-t/tf)
    
    # Vectorize pulse model function
    Mvec = np.vectorize(M)
    
    # Normalize coeff (= max M)
    norm = M(tr*np.log(tf/tr+1))
    
    return np.asarray(pha)[np.newaxis].T*Mvec(ts)/norm

def white(sigma=3e-6, mean=0.0, points=1024, t=2e-3):
    """
    Generate Gaussian White Noise
    
    Parameters (and their default values):
        sigma:  noise level in V/srHz (or gaussian sigma) (Default: 3 uV/srHz)
        mean:   DC offset of noise signal (Default: 0 V)
        t:      sample time (Default: 2 ms)
        points: data length (scalar or 2-dim tuple/list) (Default: 1024)
        points: 
    """
    
    points = np.asarray(points)
    
    # Data length
    l = points if points.ndim == 0 else points[-1]
    
    # Time resolution of Nyquist frequency
    dt = t / l * 2

    return (mean + sigma*norm.rvs(size=points)) / np.sqrt(dt)