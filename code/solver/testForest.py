import numpy as np
import sys
import forest as fr

sys.path.append('../model/')
from simulatorTLKT import Simulator
from weatherTLKT import Weather

# -----------------download--------------------------
# mydate='20171212'
# website='http://nomads.ncep.noaa.gov:9090/dods/'
# #gfs_0p25/gfs20171014/gfs_0p25_00z'
# modelcycle='00'
# resolution='1p00'
# latBound=[43,50]
# lonBound=[-10+360, 360]
# url = (website + 'gfs_' + resolution + '/gfs' + mydate + '/gfs_' + resolution + '_' + modelcycle + 'z')
# pathToSaveObj = ('../data/' + mydate + '_gfs_' + resolution[1] + '.obj')
# print(url)
# #launch in real python console
# Weather.download(url, pathToSaveObj, latBound=latBound, lonBound=lonBound, timeSteps=[0,81])

wavg = Weather.load("../data/20171212_gfs_p.obj")
# We shift the times so that all times are in the correct bounds for interpolations
wavg.time = wavg.time - wavg.time[0]

# We set up the parameters of the simulation
HOURS_TO_DAY = 1 / 24
Tf = 24 * 5
timeStep = 6  # in hour
times = np.arange(0, Tf * HOURS_TO_DAY, timeStep * HOURS_TO_DAY)
lats = np.arange(wavg.lat[0], wavg.lat[-1], 0.05)
lons = np.arange(wavg.lon[0], wavg.lon[-1], 0.05)
stateInit = [0, 47.5, -3.5 + 360]

# TODO a changer ici on donne le même simu a tous!!
# We create N simulators
N = 4
frequency = 10  # frequency of the buffer
sim = Simulator(times, lats, lons, wavg, stateInit)
sims = [sim for _ in range(N)]

stateInit = [0, 47.5, -3.5 + 360]
destination = [48, -3 + 360]
timemin = 1.8

forest = fr.Forest(listsimulators=sims, destination=destination, timemin=timemin)

forest.launch_search(stateInit, frequency)
print(forest.master.probability)
