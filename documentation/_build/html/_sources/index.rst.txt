.. IBOAT RL documentation master file, created by
   sphinx-quickstart on Sat Nov 11 18:59:10 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to IBOAT MCTS's documentation!
======================================

Context
--------
In this project we develop **a reinforcement learning algorithm** based on **parallel Monte-Carlo tree search**
to tackle the problem of **long-term path planning under uncertainty** for offshore sailing. This domain of application
is challenging as it combines unreliable and time varying wind conditions with complex and uncertain boat performances.
Contrarily to state of the art approaches applied to the sailing problem, we build a generator that models state transitions
considering these two types of uncertainty. The first one is on the boat dynamics : given a environment state the boat performances
are not deterministic. And the second one is uncertain weather forecasts. In practice, the wind is estimated from multiple
weather forecasts (each one of them being a weather scenario with a given probability of happening). The boat’s dynamics are evaluated with
a noisy Velocity Prediction Program (VPP). Then, a Monte Carlo Tree Search (MCTS) algorithm is applied in parallel to all the weather
scenarios to find the sequence of headings that minimizes the travel time between two points.


Here you will find the documentation of the **parallel MCTS** and the **isochrones** routing methods. But also all the tools to
**download and visualize weather forecasts**.

Requirements
---------------

The project depends on the following extensions :

1. NumPy for the data structures (http://www.numpy.org)
2. SciPy for interpolation (https://docs.scipy.org/doc/scipy-0.16.1/reference/index.html)
3. Matplotlib for the visualisation (https://matplotlib.org)
4. Basemap for map projections (http://matplotlib.org/basemap)
5. NetCF4 to read and write in netCDF file (https://unidata.github.io/netcdf4-python/)



Contents
---------------
.. toctree::
   :maxdepth: 2

   model <model.rst>

   solver <solver.rst>

   isochrones <isochrones.rst>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
