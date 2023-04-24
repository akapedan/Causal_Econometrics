"""
Data Analytics II: Simulation Study.

Author: Arbian Halilaj, 16-609-828.

Spring Semester 2022.

University of St. Gallen.
"""
# load the required functions
import os
from numpy.random import seed

#############################################################################

os.chdir("/Users/arbiun/Desktop/MECONI/3. Semester/Data Analytics II/Self Study/Code")

# Load own functions
import functions as pc

seed(20212022) # to ensure replicability

# Set parameters
n_sim = 1000
dim_x = 10
n = 1000

# Run simulation
results = pc.simulation(n_sim, n, dim_x)

# Show results
pc.plot_results(results,1,-1) 
pc.print_results(results,1,-1)

pc.plot_results(results,2,-1)
pc.print_results(results,2,-1)

pc.plot_results(results,3,1.5)
pc.print_results(results,3,1.5)

