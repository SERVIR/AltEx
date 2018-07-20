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

def groupObs(series,position,times,start_date,end_date):
    uniqVals = np.unique(position)
    obs = []
    dates = []
    start_date = calendar.timegm(start_date.utctimetuple()) * 1000
    end_date = calendar.timegm(end_date.utctimetuple()) * 1000
    for i in range(uniqVals.size):
        key = int(uniqVals[i])
        if times[key] != None:
            if start_date <= times[key] <= end_date:
                dates.append(times[key])
        obs.append(np.mean(series[np.where(position==key)]))


    return zip(*[dates,obs])

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

    print(iqr, lowerBound,upperBound)

    mask = np.where((series >= lowerBound) & (series <= upperBound))

    return series[mask], position[mask]


def cleanData(series,position):
    nanMask = np.where(np.isnan(series)==False)
    return series[nanMask],position[nanMask]


def outlierRemoval(series, position, clusterRange=5, interCluster=0.3):
    series, position = cleanData(series, position)
    # print 1,series,position
    heights, position = iqrFilter(series, position)
    # print 2,heights,position

    diff = np.ptp(heights, axis=0)
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

            if len(class1[0]) < len(class2[0]):
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

            print len(heights)

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
    print '------------------------Start of file:'+str(file)+'------------------------------'
    data = Dataset(file)

    # time = data.variables['time'][:]

    time_20hz = data.variables['time_20hz'][:].ravel()
    lon_20hz = data.variables['lon_20hz'][:].ravel()
    lat_20hz = data.variables['lat_20hz'][:].ravel()
    lon = data.variables['lon'][:].ravel()
    lat = data.variables['lat'][:].ravel()

    dim20Hz = lat_20hz.size

    newLat = stretchDim(lat,dim20Hz,'linear')

    print 'LatDiff: {0}'.format(np.mean(lat_20hz-newLat))

    if float(lat_range[0]) > float(lat_range[1]):
        tmp = lat_range[0]
        lat_range[0] = lat_range[1]
        lat_range[1] = tmp

    latidx = np.where((newLat >= float(lat_range[0])) & (newLat <= float(lat_range[1])))
    latidx_2d = np.where((lat_20hz >= float(lat_range[0])) & (lat_20hz <= float(lat_range[1])))
    print 'Latidx before processing',latidx
    print 'Latidx 2d',latidx_2d[0]
    # if len(latidx[0]) == 0:
    #     offset = latidx_2d[0]%20
    #     # print 'Offset',offset
    #     # print 'Latidx 2d',latidx_2d[0]
    #
    #     if 0 in offset:
    #         latidx = np.array([latidx_2d[0][offset.tolist().index(0)] / 20])
    #     else:
    #         latidx = np.array([(latidx_2d[0][0] - offset[0]) / 20])


    if len(latidx_2d) > 0:
        # latidx_2d = latidx_2d[0]

        # 1Hz VARIABLES
        alt_state_flag_ku_band_status = stretchDim(data.variables['alt_state_flag_ku_band_status'],dim20Hz,'nearest')[latidx_2d]
        model_dry_tropo_corr = stretchDim(data.variables['model_dry_tropo_corr'],dim20Hz,'nearest')[latidx_2d]
        model_wet_tropo_corr = stretchDim(data.variables['model_wet_tropo_corr'],dim20Hz,'nearest')[latidx_2d]
        iono_corr_gim_ku = stretchDim(data.variables['iono_corr_gim_ku'],dim20Hz,'nearest')[latidx_2d]
        solid_earth_tide = stretchDim(data.variables['solid_earth_tide'],dim20Hz,'nearest')[latidx_2d]
        pole_tide = stretchDim(data.variables['pole_tide'],dim20Hz,'nearest')[latidx_2d]

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

        print 'Len of alt 20hz',len(alt_20hz)
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

        print 'Length of height',len(hght)
        print 'Corrected height list:',hght
        # Check to make sure that the distance between is the points is sufficient
        # if len(latd) > 0:
        if len(hght) > 0:
            epoch = datetime.datetime(1970,1,1,0,0,0,0)
            ref = datetime.datetime(2000,1,1,0,0,0,0)
            trackTime = time_20hz[latidx_2d].mean()

            time = ref + datetime.timedelta(seconds=trackTime)

            print(time)
            time_stamp = (time-epoch).total_seconds() * 1000

            ht = np.array(hght)
            tIndex = np.full(ht.shape, int(counter))
            print '-------------------------------End of file---------------------------------'
            return int(time_stamp), ht.astype(np.float), tIndex.astype(np.uint)  # Timestamp and height for Highcharts
        else:
            return None

def calc_jason_ts(lat1,lat2,start_date,end_date,track,sensor):
    # Extract the height and timestep from a filtered netCDF file
    ts_plot = []

    height = []
    testun = []
    dt = []
    counter = 0
    gHtArray = np.array([])
    testHt = []
    testT = []
    gTArray = np.array([])
    fList = []
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
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

                        # try:
                        args = {'file': os.path.join(working_dir,file), 'lat_range': [lat1,lat2], 'counter': counter}
                        results = parse_netCDF(args)
                        if results:
                            # hgt,pos = outlierRemoval(results[1],results[2])
                            testerOut = filter_outlier(results)
                            hout = np.mean(testerOut[0])
                            if np.isnan(hout) == False:
                                testun.append(testerOut)


                                dt.append(results[0])
                                # test = np.append(test,hgt.mean())
                                gHtArray = np.append(gHtArray,results[1])
                                gTArray = np.append(gTArray, results[2])

                                testHt.append(np.mean(testerOut[0]))
                                testT.append(testerOut[1])
                                fList.append([os.path.join(working_dir,file),[lat1,lat2],counter])
                                counter += 1

                            # except Exception: # If it returns NULL, move on to the next file.
                            #     continue


                        # ts_plot.append([time_stamp,round(float(hgt),3)]) # Return this to the frontend



    # try:
    gHtArray = gHtArray.astype(np.float)
    gTArray = gTArray.astype(np.uint)
    testHt = np.array(testHt,dtype=np.float)
    # except ValueError:
    #     # pass
    #     outH = []
    #     outT = []
    #     for i in range(gHtArray.size):
    #         try:
    #             outH.append(np.float(gHtArray[i]))
    #             outT.append(np.uint(gTArray[i]))
    #         except:
    #             pass
    #     gHtArray = np.array(outH,dtype=np.float)
    #     gTArray = np.array(outT,dtype=np.uint)

    if len(dt) > 0:
        # print 'Original',gHtArray,gTArray
        print 'Raw data',dt
        print 'Len raw data',len(dt)
        series, position = outlierRemoval(gHtArray, gTArray)
        print 'After Outlier removal',series,position
        print 'Len after outlier removal',len(series),len(position)
        ts_plot = groupObs(series,position,dt,start_date,end_date)
        print 'Difference of outlier removal:',len(gHtArray) - len(series),len(gTArray) - len(position)
        print 'Percentage of original:',float(float(len(series))/float(len(gHtArray)))
        print 'Final processed',ts_plot
        # htFlat = [val for sublist in testHt for val in sublist]
        # tFlat = [val for sublist in testT for val in sublist]
        # htFlat,tFlat = iqrFilter(np.array(htFlat), np.array(tFlat))
        finalH, finalT = iqrFilter(np.array(testHt),np.array(dt))
        ts_plot2 = zip(*[finalT,finalH])#groupObs(np.array(htFlat), np.array(tFlat), dt, start_date, end_date)
        print 'After new removal process',ts_plot2
        print 'Final len:',len(ts_plot),len(ts_plot2)

        print 'Len of files processed:',len(fList)
        with open("plot.csv", 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            wr.writerows(ts_plot)
            # wr.writerows(ts_plot2)
        with open("plot2.csv", 'wb') as resultFile:
            wr = csv.writer(resultFile, dialect='excel')
            # wr.writerows(ts_plot)
            wr.writerows(ts_plot2)
        # print "Min:{0}\tMax:{1}\tMean:{2}\tStd:{3}".format(np.nanmin(test),np.nanmax(test),np.nanmean(test),np.nanstd(test))
        # with open('/home/dev/altex.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([track,start_date,end_date,len(gHtArray),len(series),len(gHtArray) - len(series),float(float(len(series))/float(len(gHtArray))),len(ts_plot),len(fList)])
    return ts_plot

t1 = datetime.datetime.now()

# calc_jason_ts('10.421329344937973','10.580663601863193','2008-01-01','2016-09-15','135','jason2')
# calc_jason_ts('12.002012','12.019098','2008-01-01','2017-12-01','140','jason2')
calc_jason_ts('12.515050883512345','12.52665571422412','2008-01-01','2017-12-01','001','jason2')

print "Processing time: {}".format(datetime.datetime.now()-t1)
