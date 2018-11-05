import numpy as np
from netCDF4 import Dataset
import numpy.ma as ma
import datetime
from scipy import interpolate
from scipy.io import *
from scipy import stats
import time
import json
import inspect
import scipy.io
import netCDF4,os,sys,glob,scipy,zipfile
import calendar
import pickle
import pandas as pd
from sklearn.cluster import KMeans
import csv
from config import JASON2_DIR,JASON3_DIR

global THREAD_POOL

geoid_grid_file = inspect.getfile(inspect.currentframe()).replace('jason.py','public/data/geoidegm2008grid.mat')
x = loadmat(geoid_grid_file)
iy=interpolate.interp2d(x['lonbp'],x['latbp'],x['grid'])

# Geoidal Correction factor. The average between the lower and upper bounds.
def geoidalCorrection(lon_,lat_):
    corr = iy(lon_, lat_)[0]
    return corr

def groupObsNew(series,position,times,start_date,end_date):
    uniqVals = np.unique(position)
    obs = []
    dates = []
    start_date = calendar.timegm(start_date.utctimetuple()) * 1000
    end_date = calendar.timegm(end_date.utctimetuple()) * 1000
    series = np.array(series)
    position = np.array(position)
    for i in range(series.size):
        key = int(uniqVals[i])
        if times[i] != None:
            if start_date <= times[i] <= end_date:
                dates.append(times[i])
        obs.append(np.mean(series[i]))

    return zip(*[dates,obs])


def iqrFilter(series, position):
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)

    iqr = stats.iqr(series)

    lowerBound = q1 - (iqr * 1.5)
    upperBound = q3 + (iqr * 1.5)


    mask = np.where((series >= lowerBound) & (series <= upperBound))

    return series[mask], position[mask]


def cleanData(series,position):
    nanMask = np.where(np.isnan(series)==False)
    return series[nanMask],position[nanMask]


def filter_outlier(results):
    Range = len(results[1])

    r = np.ptp(results[1], axis=0)
    heights = results[1]
    position = results[2]
    diff = r
    labels = None

    if heights.size > 4:

        while (diff > 5) & (len(heights)>=2):
            kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

            X = np.vstack([heights.ravel(), heights.ravel()]).T

            kmeans.fit(X)
            clusters = kmeans.cluster_centers_.squeeze()[:, 0]

            labels = kmeans.labels_

            class1 = np.where(labels == 0)
            class2 = np.where(labels == 1)

            # if clusters[0] <= clusters[1]:
            if len(class1[0]) > len(class2[0]):
                idx = class1
            else:
                idx = class2

            heights = heights[idx]
            position = position[idx]
            diff = np.abs(clusters[0] - clusters[1])

        clusterMean = heights.mean()

        std = heights.std()

        while std > 0.3:
            dist = np.abs(heights - clusterMean)
            furthestIdx = np.argmax(dist)
            keep = np.where(heights != heights[furthestIdx])
            heights = heights[keep]
            position = position[keep]

            std = heights.std()


        heights,position = iqrFilter(heights, position)
        kmeans = None

    return heights,position


def stretchDim(oldArr,newDim,type='nearest'):
    oldx = np.arange(oldArr.size)
    newx = np.linspace(0,oldx.max(),newDim)
    fInterp = interpolate.interp1d(oldx,oldArr,kind=type,fill_value='extrapolate')
    return fInterp(newx)


def parse_netCDF(args):

    file, lat_range, counter = args['file'], args['lat_range'], args['counter']
    data = Dataset(file)

    # time = data.variables['time'][:]

    time_20hz = data.variables['time_20hz'][:].ravel()
    lon_20hz = data.variables['lon_20hz'][:].ravel()
    lat_20hz = data.variables['lat_20hz'][:].ravel()
    lon = data.variables['lon'][:].ravel()
    lat = data.variables['lat'][:].ravel()

    dim20Hz = lat_20hz.size

    newLat = stretchDim(lat,dim20Hz,'linear')

    if float(lat_range[0]) > float(lat_range[1]):
        tmp = lat_range[0]
        lat_range[0] = lat_range[1]
        lat_range[1] = tmp

    latidx = np.where((newLat >= float(lat_range[0])) & (newLat <= float(lat_range[1])))
    latidx_2d = np.where((lat_20hz >= float(lat_range[0])) & (lat_20hz <= float(lat_range[1])))

    if len(latidx_2d) > 0:
        # 1Hz VARIABLES
        alt_state_flag_ku_band_status = stretchDim(data.variables['alt_state_flag_ku_band_status'],dim20Hz,'nearest')[latidx_2d]
        model_dry_tropo_corr = stretchDim(data.variables['model_dry_tropo_corr'],dim20Hz,'linear')[latidx_2d]
        model_wet_tropo_corr = stretchDim(data.variables['model_wet_tropo_corr'],dim20Hz,'linear')[latidx_2d]
        iono_corr_gim_ku = stretchDim(data.variables['iono_corr_gim_ku'],dim20Hz,'linear')[latidx_2d]
        solid_earth_tide = stretchDim(data.variables['solid_earth_tide'],dim20Hz,'linear')[latidx_2d]
        pole_tide = stretchDim(data.variables['pole_tide'],dim20Hz,'linear')[latidx_2d]

        # 20Hz VARIABLES
        alt_20hz = data.variables['alt_20hz'][:].ravel()[latidx_2d]
        ice_range_20hz_ku = data.variables['ice_range_20hz_ku'][:].ravel()[latidx_2d]
        ice_sig0_20hz_ku = data.variables['ice_sig0_20hz_ku'][:].ravel()[latidx_2d]
        ice_qual_flag_20hz_ku = data.variables['ice_qual_flag_20hz_ku'][:].ravel()[latidx_2d]

        lons = lon_20hz[latidx_2d]
        lats = lat_20hz[latidx_2d]

        time_20hz_units = data.variables['time_20hz'].units

        dim_size = alt_state_flag_ku_band_status.size

        mjd_20hz = []
        hght = []
        longt = []
        latd = []
        bs = []

        if len(alt_state_flag_ku_band_status) > 0:
            if len(alt_20hz) > len(alt_state_flag_ku_band_status)*20:
                leftover = leftover + 20

        for i in range(0, dim_size):

            # A check to see if the alt state flag ku band status is good
            if alt_state_flag_ku_band_status[i] != 0:
                continue

            media_corr = model_dry_tropo_corr[i] + model_wet_tropo_corr[i] + iono_corr_gim_ku[i] + solid_earth_tide[i] + \
                         pole_tide[i]

            if ice_qual_flag_20hz_ku[i] != 0:
                continue

            gc = geoidalCorrection(lons[i], lats[i])
            hght.append(alt_20hz[i] - (media_corr + ice_range_20hz_ku[i]) - gc - 0.7)
            bs.append(ice_sig0_20hz_ku[i])

        if len(hght) > 0:
            epoch = datetime.datetime(1970,1,1,0,0,0,0)
            ref = datetime.datetime(2000,1,1,0,0,0,0)
            trackTime = time_20hz[latidx_2d].mean()

            time = ref + datetime.timedelta(seconds=trackTime)

            time_stamp = (time-epoch).total_seconds() * 1000

            ht = np.array(hght)
            tIndex = np.full(ht.shape, int(counter))
            return int(time_stamp), ht.astype(np.float), tIndex.astype(np.uint)  # Timestamp and height for Highcharts
        else:
            return None

def calc_jason_ts(lat1,lat2,start_date,end_date,track,sensor):
    # Extract the height and timestep from a filtered netCDF file
    ts_plot = []

    height = []
    dt = []
    counter = 0
    testHt = []
    fList = []

    try:
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except:
        pass

    f_start_date = None
    f_end_date = None
    JASON_DIR = None
    if sensor == 'jason2':
        JASON_DIR = JASON2_DIR
        f_start_date = datetime.datetime.strptime('2008-01-01', "%Y-%m-%d")
        f_end_date = datetime.datetime.strptime('2018-01-01', "%Y-%m-%d")
    elif sensor == 'jason3':
        JASON_DIR = JASON3_DIR
        f_start_date = datetime.datetime.strptime('2017-01-01', "%Y-%m-%d")
        f_end_date = datetime.datetime.strptime('2020-01-01', "%Y-%m-%d")
    for dir in sorted(os.listdir(JASON_DIR)):
        working_dir = os.path.join(JASON_DIR,dir)

        for file in sorted(os.listdir(working_dir)):
            if track in file:

                file_name = file[12:]
                file_info = file_name.split('_')
                cycle_num = file_info[0]
                pass_num = file_info[1]

                pass_start_date = file_info[2]+'_'+file_info[3]
                pass_end_date = file_info[4]+'_'+file_info[5][:-3]

                pass_start_date = datetime.datetime.strptime(pass_start_date, "%Y%m%d_%H%M%S")
                pass_end_date = datetime.datetime.strptime(pass_end_date, "%Y%m%d_%H%M%S")

                if track == pass_num:
                    if f_start_date <= pass_start_date <= pass_end_date <= f_end_date:

                        args = {'file': os.path.join(working_dir,file), 'lat_range': [lat1,lat2], 'counter': counter}
                        results = parse_netCDF(args)
                        #print results
                        try:
                            testerOut = filter_outlier(results)
                        except:
                            pass
                        if results:
                            hout = np.mean(testerOut[0])
                            if np.isnan(hout) == False:
                                dt.append(results[0])
                                testHt.append(np.mean(testerOut[0]))
                                fList.append([os.path.join(working_dir,file),[lat1,lat2],counter])
                                counter += 1

    testHt = np.array(testHt,dtype=np.float)

    if len(dt) > 0:

        finalH, finalT = iqrFilter(np.array(testHt),np.array(dt))
        ts_plot = zip(*[finalT,finalH])

    return ts_plot

# t1 = datetime.datetime.now()

# Niger
# calc_jason_ts('10.421329344937973','10.580663601863193','2008-01-01','2017-12-31','135','jason2')
# Myanmar
# calc_jason_ts('16.147471112269287','16.22330557626728','2008-01-01','2017-12-01','205','jason2')
# cambodia
# calc_jason_ts('12.51562944586206','12.525013877044515','2008-01-01','2017-12-15','001','jason2')

# print "Processing time: {}".format(datetime.datetime.now()-t1)
