"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import copy
import numpy as np

config = {
    'jobname': 'MPS_simulation',  # name of the job
    'task': {
        'type': 'PythonFunctionCall',
        'module': 'simulation',
        'function': 'simulation'
    },
    'task_parameters': [],  # list of dict containing the **kwargs given to the `function`
    'requirements_slurm': {  # passed on to SLURM
        'time': '2-00:00:00',  # d-hh:mm:ss
        'mem': '1G',
        'partition': 'cpu',
        'qos': 'normal',
        'nodes': 1,  # number of nodes
        'cpus-per-task': 6,  # number of CPUs per task
    },
    'options': {
    }
}

sigmay = np.array([[0, -1j], [1j, 0]])
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.array([[1.0, 0.0], [0.0, -1.0]])

for chi in [100,200]:

    kwargs = {
        'L': 70,
        'J': 1.0,
        'g': 0.15,
        'X': sigmay,
        'Y': sigmay,
        'n': 12,
        'dt': 0.01,
        'k': 0.1,
        'h': 0,
        'chi_max': chi,
        'site': None,
        'savedir': "simulations",
        'ops_name': "sigmay"
    }

    config['task_parameters'].append(copy.deepcopy(kwargs))


cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
