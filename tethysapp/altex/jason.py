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
import multiprocessing as mp

global THREAD_POOL

# Replace this with the path to the data directory
JASON_DIR = '/home/dev/avisoftp.cnes.fr/AVISO/pub/jason-2/gdr_d'

geoid_grid_file = inspect.getfile(inspect.currentframe()).replace('jason.py','public/data/geoidegm2008grid.mat')
x = loadmat(geoid_grid_file)
iy=interpolate.interp2d(x['lonbp'],x['latbp'],x['grid'])

# Geoidal Correction factor. The average between the lower and upper bounds.
def geoidalCorrection(lon_,lat_):
    corr = iy(lon_, lat_)[0]
    return corr

def groupObs(series,position,times):
    uniqVals = np.unique(position)
    obs = []
    dates = []

    for i in range(uniqVals.size):
        key = int(uniqVals[i])

        if times[key] != None:
            dates.append(times[key])
        obs.append(np.mean(series[np.where(position==key)]))


    return zip(*[dates,obs])


def iqrFilter(series, position):
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)

    iqr = q3 - q1

    lowerBound = q1 - (iqr * 1.5)
    upperBound = q3 + (iqr * 1.5)

    mask = np.where((series > lowerBound) & (series < upperBound))

    return series[mask], position[mask]


def cleanData(series,position):
    nanMask = np.where(np.isnan(series)==False)
    return series[nanMask],position[nanMask]


def outlierRemoval(series, position, clusterRange=5, interCluster=0.3):
    series, position = cleanData(series, position)

    heights, position = iqrFilter(series, position)

    diff = 10
    labels = None

    if heights.size > 5:

        while diff > clusterRange:
            kmeans = KMeans(init='k-means++', n_clusters=2, n_init=20)

            X = np.vstack([heights.ravel(), heights.ravel()]).T

            kmeans.fit(X)
            clusters = kmeans.cluster_centers_.squeeze()[:, 0]

            diff = np.abs(clusters[0] - clusters[1])

            labels = kmeans.labels_

            class1 = np.where(labels==0)
            class2 = np.where(labels==1)

            if len(class1[0]) > len(class2[0]):
                idx = class1
            else:
                idx = class2

            heights = heights[idx]
            position = position[idx]



        if len(class1[0]) > len(class2[0]):
            clusterMean = clusters[0]
        else:
            clusterMean = clusters[1]

        std = 1

        while std > interCluster:
            dist = np.abs(heights - clusterMean)
            furthestIdx = np.argmax(dist)
            keep = np.where(heights != heights[furthestIdx])
            heights = heights[keep]
            position = position[keep]

            std = heights.std()

        kmeans = None
        finalSeries, finalPosition = iqrFilter(heights, position)

    else:
        finalSeries, finalPosition = heights, position

    return finalSeries, finalPosition

def parse_netCDF(args):
    file, lat_range, counter = args['file'], args['lat_range'], args['counter']

    data = Dataset(file)
    nc_dims = [dim for dim in data.dimensions]

    hz = len(data.dimensions[nc_dims[1]])
    time = data.variables['time'][:]

    time_20hz = data.variables['time_20hz'][:]
    lon_20hz = data.variables['lon_20hz'][:]
    lat_20hz = data.variables['lat_20hz'][:]
    lon = data.variables['lon'][:]
    lat = data.variables['lat'][:]

    lat1_idx = (np.abs(lat - float(lat_range[0]))).argmin()
    lat2_idx = (np.abs(lat - float(lat_range[1]))).argmin()

    if lat2_idx < lat1_idx:
        tmp = lat1_idx
        lat1_idx = lat2_idx
        lat2_idx = tmp

    alt_state_flag_ku_band_status = data.variables['alt_state_flag_ku_band_status'][lat1_idx:lat2_idx]
    alt_20hz = data.variables['alt_20hz'][lat1_idx:lat2_idx]
    model_dry_tropo_corr = data.variables['model_dry_tropo_corr'][lat1_idx:lat2_idx]
    model_wet_tropo_corr = data.variables['model_wet_tropo_corr'][lat1_idx:lat2_idx]
    iono_corr_gim_ku = data.variables['iono_corr_gim_ku'][lat1_idx:lat2_idx]
    solid_earth_tide = data.variables['solid_earth_tide'][lat1_idx:lat2_idx]
    pole_tide = data.variables['pole_tide'][lat1_idx:lat2_idx]
    ice_range_20hz_ku = data.variables['ice_range_20hz_ku'][lat1_idx:lat2_idx]
    ice_sig0_20hz_ku = data.variables['ice_sig0_20hz_ku'][lat1_idx:lat2_idx]
    ice_qual_flag_20hz_ku = data.variables['ice_qual_flag_20hz_ku'][lat1_idx:lat2_idx]
    lons = lon_20hz[lat1_idx:lat2_idx]
    lats = lat_20hz[lat1_idx:lat2_idx]

    time_20hz_units = data.variables['time_20hz'].units

    dim_size = alt_state_flag_ku_band_status.size

    mjd_20hz = []
    hght = []
    longt = []
    latd = []
    bs = []

    for i in range(0, dim_size):

        # A check to see if the alt state flag ku band status is good
        if alt_state_flag_ku_band_status[i] != 0:
            continue

        media_corr = model_dry_tropo_corr[i] + model_wet_tropo_corr[i] + iono_corr_gim_ku[i] + solid_earth_tide[i] + \
                     pole_tide[i]

        for j in range(0, hz):

            if ice_qual_flag_20hz_ku[i, j] != 0:
                continue

            if lon_20hz[i, j] == 'nan':
                continue

            mjd_20hz.append(time_20hz[i, j])
            latd.append(lat_20hz[i, j])
            longt.append(lon_20hz[i, j])
            gc = geoidalCorrection(lons[i, j], lats[i, j])

            hght.append(alt_20hz[i, j] - (media_corr + ice_range_20hz_ku[i, j]) - gc - 0.7)
            bs.append(ice_sig0_20hz_ku[i, j])

    # Check to make sure that the distance between is the points is sufficient
    if len(latd) > 0:

        mjd_20hz = (np.array(mjd_20hz)).mean()

        # At times the date is out of range, a hack to get around that.
        try:
            mjd = netCDF4.num2date(mjd_20hz, time_20hz_units, calendar='gregorian')
        except Exception:
            return None

        time_stamp = calendar.timegm(mjd.utctimetuple()) * 1000  # Converting date to utc timestamp for Highcharts
        ht = np.array(hght)
        tIndex = np.full(ht.shape, int(counter))

        return int(time_stamp), ht.astype(np.float), tIndex.astype(np.uint)  # Timestamp and height for Highcharts
    else:
        return None

def calc_jason_ts(lat1,lon1,lat2,lon2,start_date,end_date,track):
    # Extract the height and timestep from a filtered netCDF file
    ts_plot = []

    height = []
    dt = []
    counter = 0
    gHtArray = np.array([])
    gTArray = np.array([])
    fList = []
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
                    if start_date <= pass_start_date <= pass_end_date <= end_date:

                        try:
                            args = {'file': os.path.join(working_dir,file), 'lat_range': [lat1,lat2], 'counter': counter}
                            results = parse_netCDF(args)
                            dt.append(results[0])
                            gHtArray = np.append(gHtArray,results[1])
                            gTArray = np.append(gTArray,results[2])
                            # fList.append([os.path.join(working_dir,file),[lat1,lat2],counter])
                            counter +=1
                        except Exception: # If it returns NULL, move on to the next file.
                            continue
                        # ts_plot.append([time_stamp,round(float(hgt),3)]) # Return this to the frontend

    # ncores = mp.cpu_count()
    # global THREAD_POOL
    #
    # if ncores < 4:
    #     THREAD_POOL = ncores
    #
    # elif ncores < 8:
    #     THREAD_POOL = 4
    #
    # elif ncores < 16:
    #     THREAD_POOL = 12
    #
    # else:
    #     THREAD_POOL = int(ncores / 0.7)
    #
    # pool = mp.Pool(THREAD_POOL)
    # args = [{'file': i[0], 'lat_range': i[1], 'counter': i[2]} for i in fList]
    # results = pool.map(parse_netCDF,args)
    # # pool.close()
    # # pool.join()
    # dt,gHtArray,gTArray = [], np.array([]) ,np.array([])
    #
    # for i in results:
    #     if i != None:
    #         dt.append(i[0])
    #         gHtArray = np.append(gHtArray,i[1])
    #         gTArray = np.append(gTArray,i[2])
    #     else:
    #         dt.append(None)

    try:
        gHtArray = gHtArray.astype(np.float)
        gTArray = gTArray.astype(np.uint)
    except ValueError:
        outH = []
        outT = []
        for i in range(gHtArray.size):
            try:
                outH.append(np.float(gHtArray[i]))
                outT.append(np.uint(gTArray[i]))
            except:
                pass
        gHtArray = np.array(outH,dtype=np.float)
        gTArray = np.array(outT,dtype=np.uint)

    if len(dt) > 0:
        series, position = outlierRemoval(gHtArray, gTArray)
        ts_plot = groupObs(series,position,dt)

    return ts_plot

#calc_jason_ts('6.567022920438291','-0.22796630859374875','6.70343214627151','-0.27740478515624895','2008-01-01','2008-12-01','046')
# calc_jason_ts('12.002012','105.481','12.019098','105.481','2008-01-01','2008-12-01','140')
# calc_jason_ts('12.492662','104.483','12.533555','104.483','2008-01-01','2008-12-01','001')