#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:46:52 2017

@author: paul
"""


import pygrib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

plt.figure()
    
grib='myTestGribFile.grb';
grbs=pygrib.open(grib)

#In this example we will extract the same significant wave height field we used in the first example. Remember that indexing in Python starts at zero.

grb = grbs.select(name='Significant height of wind waves')[0]
data=grb.values
lat,lon = grb.latlons()



#From this point on the code is almost identical to the previous example.
#
#Plot the field using Basemap. Start with setting the map projection using the limits of the lat/lon data itself:

m=Basemap(projection='mill',lat_ts=10,llcrnrlon=lon.min(), \
urcrnrlon=lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
      resolution='c')

#Convert the lat/lon values to x/y projections.

x, y = m(lon,lat)

#Next, plot the field using the fast pcolormesh routine and set the colormap to jet.

cs = m.pcolormesh(x,y,data,shading='flat',cmap=plt.cm.jet)

#Add a coastline and axis values.

m.drawcoastlines()
m.fillcontinents()
m.drawmapboundary()
m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0])
m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])

#Add a colorbar and title, and then show the plot.

plt.colorbar(cs,orientation='vertical')
plt.title('Example 2: NWW3 Significant Wave Height from GRiB')
plt.show()

