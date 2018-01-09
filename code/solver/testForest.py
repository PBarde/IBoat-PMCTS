import numpy as np
import sys
import forest as fr

sys.path.append('../model/')
from simulatorTLKT import Simulator
from weatherTLKT import Weather

# -----------------download--------------------------
mydate = '20180103'
website = 'http://nomads.ncep.noaa.gov:9090/dods/'
# gfs_0p25/gfs20171014/gfs_0p25_00z'
modelcycle = range(1, 21)
# resolution = '1p00'
latBound = [43, 50]
lonBound = [-10 + 360, 360]
pathToSaveObj = []
for ii in modelcycle:
    # url = (website + 'gfs_' + resolution + '/gfs' + mydate + '/gfs_' + resolution + '_' + modelcycle + 'z')
    if ii < 10:
        cycle = '0' + str(ii)
    else:
        cycle = str(ii)

    url = (website + 'gens/gens' + mydate + '/gep' + cycle + '_00z')
    pathToSaveObj.append(('../data/' + mydate + '_gep_' + cycle + '00z.obj'))

    # Uncomment the following lines to download the scenarios (launch in real python console)
    # print(url)
    # Weather.download(url, pathToSaveObj[ii-1], latBound=latBound, lonBound=lonBound, timeSteps=[0, 64], ens=True)

#%%
# We create N simulators based on the scenarios
N = 1  # <=20
frequency = 10  # frequency of the buffer
stateInit = [0, 47.5, -3.5 + 360]
destination = [48, -3 + 360]
sims = []
for jj in range(N):
    weather_scen = Weather.load(pathToSaveObj[jj])
    # We shift the times so that all times are in the correct bounds for interpolations
    weather_scen.time = weather_scen.time - weather_scen.time[0]

    # We set up the parameters of the simulation
    HOURS_TO_DAY = 1 / 24
    timeStep = 6  # in hour
    timemax = timeStep * len(weather_scen.time)
    Tf = 24 * 5
    times = np.arange(0, Tf * HOURS_TO_DAY, timeStep * HOURS_TO_DAY)
    lats = np.arange(weather_scen.lat[0], weather_scen.lat[-1], 0.05)
    lons = np.arange(weather_scen.lon[0], weather_scen.lon[-1], 0.05)
    sims.append(Simulator(times, lats, lons, weather_scen, stateInit))

timemin = 1.8
forest = fr.Forest(listsimulators=sims, destination=destination, timemin=timemin)
forest.launch_search(stateInit, frequency)

forest.master.get_children()
forest.master.get_depth()
forest.master.plot_best_policy()

