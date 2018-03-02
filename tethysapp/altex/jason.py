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
from config import JASON2_DIR,JASON3_DIR

global THREAD_POOL

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
    # print 1,series,position
    heights, position = iqrFilter(series, position)
    # print 2,heights,position
    diff = 10
    labels = None

    if heights.size > 5:

        while diff > clusterRange:
            kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

            X = np.vstack([heights.ravel(), heights.ravel()]).T

            kmeans.fit(X)
            clusters = kmeans.cluster_centers_.squeeze()[:, 0]

            labels = kmeans.labels_

            class1 = np.where(labels==0)
            class2 = np.where(labels==1)

            if len(class1[0]) > len(class2[0]):
                idx = class1
            else:
                idx = class2

            heights = heights[idx]
            position = position[idx]
            diff = np.abs(clusters[0] - clusters[1])


        clusterMean = heights.mean()

        std = heights.std()

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

    # time = data.variables['time'][:]

    time_20hz = data.variables['time_20hz'][:].ravel()
    lon_20hz = data.variables['lon_20hz'][:].ravel()
    lat_20hz = data.variables['lat_20hz'][:].ravel()
    # lon = data.variables['lon'][:].ravel()
    lat = data.variables['lat'][:].ravel()

    if float(lat_range[0]) > float(lat_range[1]):
        tmp = lat_range[0]
        lat_range[0] = lat_range[1]
        lat_range[1] = tmp

    latidx = np.where((lat >= float(lat_range[0])) & (lat <= float(lat_range[1])))
    latidx_2d = np.where((lat_20hz >= float(lat_range[0])) & (lat_20hz <= float(lat_range[1])))

    if len(latidx[0]) == 0:
        offset = latidx_2d[0]%20

        if 0 in offset:
            latidx = np.array([latidx_2d[0][offset.tolist().index(0)] / 20])
        else:
            latidx = np.array([(latidx_2d[0][0] - offset[0]) / 20])

    alt_state_flag_ku_band_status = data.variables['alt_state_flag_ku_band_status'][latidx]
    alt_20hz = data.variables['alt_20hz'][:].ravel()[latidx_2d]
    model_dry_tropo_corr = data.variables['model_dry_tropo_corr'][latidx]
    model_wet_tropo_corr = data.variables['model_wet_tropo_corr'][latidx]
    iono_corr_gim_ku = data.variables['iono_corr_gim_ku'][latidx]
    solid_earth_tide = data.variables['solid_earth_tide'][latidx]
    pole_tide = data.variables['pole_tide'][latidx]
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
    cnt = 0
    idxcounter = 0

    leftover = (lats.size - cnt)%20


    if len(alt_state_flag_ku_band_status) > 0:
        if len(alt_20hz) > len(alt_state_flag_ku_band_status)*20:
            leftover = leftover + 20

    for i in range(0, dim_size):

        # A check to see if the alt state flag ku band status is good
        if alt_state_flag_ku_band_status[i] != 0:
            continue

        media_corr = model_dry_tropo_corr[i] + model_wet_tropo_corr[i] + iono_corr_gim_ku[i] + solid_earth_tide[i] + \
                     pole_tide[i]

        if i == dim_size-1:
            hz = leftover
        else:
            hz = 20

        for j in range(0, hz):

            if ice_qual_flag_20hz_ku[idxcounter] != 0:
                continue

            # if lon_20hz[i, j] == 'nan':
            #     continue

            if np.ma.is_masked(time_20hz[idxcounter]) == False:
                # print 'd2d23d;',time_20hz[idxcounter]
                mjd_20hz.append(time_20hz[idxcounter])

            gc = geoidalCorrection(lons[idxcounter], lats[idxcounter])

            hght.append(alt_20hz[idxcounter] - (media_corr + ice_range_20hz_ku[idxcounter]) - gc - 0.7)
            bs.append(ice_sig0_20hz_ku[idxcounter])
            cnt+=1
            idxcounter+=1

    # Check to make sure that the distance between is the points is sufficient
    # if len(latd) > 0:
    if len(mjd_20hz) > 0:
        mjd_20hz = np.mean(np.array(mjd_20hz))
        mjd = netCDF4.num2date(mjd_20hz, time_20hz_units, calendar='gregorian')
        time_stamp = calendar.timegm(mjd.utctimetuple()) * 1000  # Converting date to utc timestamp for Highcharts
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
    gHtArray = np.array([])
    gTArray = np.array([])
    fList = []
    # start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    # end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    JASON_DIR = None
    if sensor == 'jason2':
        JASON_DIR = JASON2_DIR
    elif sensor == 'jason3':
        JASON_DIR = JASON3_DIR

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
                            gHtArray = np.append(gHtArray, results[1])
                            gTArray = np.append(gTArray, results[2])
                            # fList.append([os.path.join(working_dir,file),[lat1,lat2],counter])
                            counter += 1
                        # break
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
        # print 'Original',gHtArray,gTArray
        series, position = outlierRemoval(gHtArray, gTArray)
        ts_plot = groupObs(series,position,dt)

    return ts_plot

# calc_jason_ts('6.567022920438291','-0.22796630859374875','6.70343214627151','-0.27740478515624895','2008-01-01','2008-12-01','046')
# calc_jason_ts('12.002012','105.481','12.019098','105.481','2008-01-01','2008-12-01','140')
# calc_jason_ts('12.515050883512345','104.47629158353095','12.52665571422412','104.48084061002022','2008-01-01','2008-12-01','001')