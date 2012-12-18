import numpy as np
from scipy.stats import norm, cauchy
from pytes import Analysis, Pulse, Constants

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
        return ((1-np.exp(-t/tr)) * np.exp(-t/tf)) * (t>0)
    
    # Normalize coeff (= max M)
    norm = M(tr*np.log(tf/tr+1))
    
    return np.asarray(pha)[np.newaxis].T*M(ts)/norm

def white(sigma=3e-6, mean=0.0, points=1024, t=2e-3):
    """
    Generate Gaussian White Noise
    
    Parameters (and their default values):
        sigma:  noise level in V/srHz (or gaussian sigma) (Default: 3 uV/srHz)
        mean:   DC offset of noise signal (Default: 0 V)
        t:      sample time (Default: 2 ms)
        points: data length (scalar or 2-dim tuple/list) (Default: 1024)
    """
    
    points = np.asarray(points)
    
    # Data length
    l = points if points.ndim == 0 else points[-1]
    
    # Time resolution of Nyquist frequency
    dt = t / l * 2

    return (mean + sigma*norm.rvs(size=points)) / np.sqrt(dt)

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
    
    maxp = np.max(pdf(np.linspace(min, max, 1e6)))
    
    while len(valid) < N:
        r = np.random.uniform(min, max, N)
        p = np.random.uniform(0, maxp, N)
        
        valid = np.concatenate((valid, r[p < pdf(r)]))
    
    return valid[:N]

def simulate(N, width, noise=3e-6, sps=1e6, t=2e-3, Emax=10e3, atom="Mn", ratio=0.9, talign=True):
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
        ratio:  Ka ratio (Default: 0.9)
        taling: vary pulse alignment (Default: True)
    
    Return (pulse, noise):
        pulse:  simulated pulse data (NxM array-like)
        noise:  simulated noise data (NxM array-like)
    
    Note:
        - pha 1.0 = Emax
    """
    
    # Simulate Ka and Kb Lines
    e = np.concatenate((line(int(N*ratio), width, atom+"Ka"),
                        line(int(N-N*ratio), width, atom+"Kb")))
    
    # Convert energy to PHA
    pha = e / Emax
    
    # Vary talign?
    if talign:
        ta = np.random.uniform(size=N) - 0.5
    else:
        ta = 0
    
    # Generate pulses and noises
    points = (N, int(sps*t))
    p = pulse(pha, points=points, t=t, duty=0.1, talign=ta) + \
                white(sigma=noise, points=points, t=t)
    
    n = white(sigma=noise, points=points, t=t)
    
    return p, n

def line(N, width, line="MnKa"):
    """
    Simulate Ka/Kb Line
    
    Parameters (and their default values):
        N:      desired number of pulse
        width:  width (FWHM) of gaussian (voigt) profile
        line:   line to simulate (Default: MnKa)
    
    Return (e):
        e:      simulated data
    """

    if width == 0:
        # Simulate by Cauchy (Lorentzian)
        e = np.array([])
        fs = np.asarray(Constants.FS[line])
        for f in fs[fs.T[2].argsort()]:
            e = np.concatenate((e, cauchy.rvs(loc=f[0], scale=Analysis.fwhm2gamma(f[1]), size=int(f[2]*N))))
        e = np.concatenate((e, cauchy.rvs(loc=f[0], scale=Analysis.fwhm2gamma(f[1]), size=int(N-len(e)))))
        
    else:
        # Simulate by Voigt
        pdf = lambda E: Analysis.line_model(E, 0, width, line=line)
        Ec = np.array(Constants.FS[line])[:,0]
        _Emin = np.min(Ec) - width*50
        _Emax = np.max(Ec) + width*50
        e = random(pdf, N, _Emin, _Emax)
    
    return e