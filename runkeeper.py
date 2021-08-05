import xml.etree.ElementTree as ET
import numpy as np
import time
import datetime
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import smopy
#import matplotlib.pyplot as plt
import image

# Radius of the earth in km
r_earth = 6378

class Activity():

    def __init__(self,gpx_file):

        tree = ET.parse(gpx_file)

        tvec = []
        lat = []
        lon = []
        ele = []
        starttime = None

        
        nTrackSegments = len(tree.getroot()[0])
        
        for trackSeg in range(nTrackSegments):
            for trackpt in tree.getroot()[0][trackSeg]:
            #for trackpt in tree.getroot()[0][2]:

                if starttime is None:
                    starttime = datetime.datetime.strptime(trackpt[1].text,'%Y-%m-%dT%H:%M:%SZ')
                tvec.append( (datetime.datetime.strptime(trackpt[1].text,'%Y-%m-%dT%H:%M:%SZ') - starttime).total_seconds() )
                lat.append(float(trackpt.get('lat')))
                lon.append(float(trackpt.get('lon')))
                ele.append(float(trackpt[0].text))


        #for i in range(len(lat)):
        #    print("lat/lon/ele: {}/{}/{}".format(lat[i], lon[i],ele[i]))
        
        self.lat = interp1d(tvec,lat)
        self.lon = interp1d(tvec,lon)
        self.ele = interp1d(tvec,ele)
        self.extent = (min(lat),min(lon),max(lat),max(lon))
        
        

        latlong = np.hstack((np.array(lon)[:,np.newaxis], np.array(lat)[:,np.newaxis])) / 180. * 3.14159
        latlong[:,0] = latlong[:,0] * np.cos(np.mean(lat) / 180 * 3.14159) # Damn spherical earth!

        traveled =  np.concatenate( (np.zeros(1), np.sqrt( np.sum( (latlong[1:,:] - latlong[:-1,:])**2,axis=1 )) )) * r_earth

        distance = cumtrapz(traveled,initial=0.)
        self.total_distance = max(distance)
        self.total_time = max(tvec)

        self.distance = interp1d(tvec,distance)
        self.starttime = time.mktime(starttime.timetuple())

        self.map = smopy.Map(self.extent)
        
        
        width, height = self.map.img.size
        maxWidth = int(1920/4)
        maxHeight = int(1080/3)
        
        #resize this image, if it's bigger than a third
        #if (width > maxWidth or height >maxHeight):
        #    ratioWidth = width/maxWidth
        #    ratioHeight = height/maxHeight
            
            #print("ratioWidth> {}".format(ratioWidth))
            #print("ratioHeight> {}".format(ratioHeight))
            
        #    if (ratioWidth > ratioHeight):
                #print("map resized to> {}".format((maxWidth, int(height/ratioWidth))))
        #        self.map.img = self.map.img.resize((maxWidth, int(height/ratioWidth)), 3) #3=bicubic filtering
        #    else:
                #print("Width: {}, maxheight> {}, width/maxHeight> {}".format(width, ratioHeight, width/ratioHeight))
                #print("map resized to> {}".format((int(width/ratioHeight), maxHeight)))
        #        self.map.img = self.map.img.resize((int(width/ratioHeight), maxHeight), 3) #3=bicubic filtering
        
        #print("map size> {}".format(self.map.img.size))
        #print("map zoom> {}".format(self.map.z))
        #exit()
        #print("map dir: {}".format(dir(self.map)))
        #ax = self.map.show_mpl(figsize=(8, 6))
        #ax.plot(self.map.xmin, self.map.ymin, 'or', ms=10, mew=2);
        #plt.show()
        #print("map extent> {}".format(self.extent))
        #stop
    
    def get_position(self,times,units='latlong'):
        '''
        Get position, based on linear interpolation between neighbouring points
        '''

        if units == 'latlong':

            return self.lat(times),self.lon(times)

        elif units == 'map_pixels':

            return self.map.to_pixels(self.lat(times),self.lon(times))


    @property
    def map_bg(self):
        return self.map.img