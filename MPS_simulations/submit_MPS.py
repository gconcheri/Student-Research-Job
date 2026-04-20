"""Example how to create a `config` for a job array and submit it using cluster_jobs.py."""

import cluster_jobs
import copy
import numpy as np  # only needed if you use np below

config = {
    'jobname': 'MyJob',
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
        'qos': 'debug',
        'nodes': 10,  # number of nodes
    },
    #  'requirements_sge': {  # for SGE
    #      'l': 'h_cpu=0:30:00,h_rss=4G',
    #      'q': 'queue',
    #      'pe smp': '4',
    #      # 'M': "no@example.com"
    #  },
    'options': {
        # you can add extra variables for the script_template in cluster_templates/* here
    }
}


kwargs = {
    'L': 55,
    'J': 1.0,
    'g': 0.15,
    'X': 'sigmay',
    'Y': 'sigmay',
    'n': 12,
    'dt': 0.01,
    'k': 0.1,
    'h': 0,
    'chi_max': 200,
    'site': None,
    'savedir': "simulations",
    'ops_name': "sigmay"
}
config['task_parameters'].append(copy.deepcopy(kwargs))


cluster_jobs.SlurmJob(**config).submit()  # submit to SLURM
