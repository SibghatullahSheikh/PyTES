# Constants
#
# Mostly taken from: 10.1103/PhysRevA.56.4554

import numpy as np

## Fine Structure
FS = {
    "MnKa": (
        (5898.853, 1.715, 0.353),   # alpha_11
        (5897.867, 2.043, 0.141),   # alpha_12
        (5894.829, 4.499, 0.079),   # alpha_13
        (5896.532, 2.663, 0.066),   # alpha_14
        (5899.417, 0.969, 0.018),   # alpha_15
        (5887.743, 2.361, 0.229),   # alpha_21
        (5886.495, 4.216, 0.110)),  # alpha_22
    "MnKb": (
        (6490.89, 1.83, 0.254),     # beta_a
        (6486.31, 9.40, 0.234),     # beta_b
        (6477.73, 13.22, 0.234),    # beta_c
        (6490.06, 1.81, 0.164),     # beta_d
        (6488.83, 2.81, 0.114))     # beta_e
}

## Line Energy
LE = {
    "Mn": (
        np.exp(np.log(np.asarray(FS["MnKa"])[:,(0,2)]).sum(axis=1)).sum(),
        np.exp(np.log(np.asarray(FS["MnKb"])[:,(0,2)]).sum(axis=1)).sum()
    )
}