try:
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import traceback
    from torch.autograd.functional import jacobian, hessian
    from torch.autograd import grad
    from IPython import display 
    from pyfiglet import Figlet
    print("All the dependecies have been found. So far so good.")
except ImportError:
    print("Some dependecies haven't been found.")


