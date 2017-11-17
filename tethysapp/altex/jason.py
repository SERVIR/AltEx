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

# Replace this with the path to the data directory
JASON_DIR = '/home/dev/avisoftp.cnes.fr/AVISO/pub/jason-2/gdr_d'

geoid_grid_file = inspect.getfile(inspect.currentframe()).replace('jason.py','public/data/geoidegm2008grid.mat')
x = loadmat(geoid_grid_file)
iy=interpolate.interp2d(x['lonbp'],x['latbp'],x['grid'])

# Geoidal Correction factor. The average between the lower and upper bounds.
def geoidalCorrection(lon_,lat_):
    corr = iy(lon_, lat_)[0]
    return corr

# Extract the height and timestep from a filtered netCDF file
def parse_netCDF(file,lat_range,geoid):

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
            hght.append(alt_20hz[i, j] - (media_corr + ice_range_20hz_ku[i, j]))
            bs.append(ice_sig0_20hz_ku[i, j])

    # Check to make sure that the distance between is the points is sufficient
    if len(latd) > 0:

        mjd_20hz = (np.array(mjd_20hz)).mean()

        # At times the date is out of range, a hack to get around that.
        try:
            mjd = netCDF4.num2date(mjd_20hz, time_20hz_units, calendar='gregorian')
        except Exception:
            return 'NULL'

        time_stamp = calendar.timegm(mjd.utctimetuple()) * 1000 # Converting date to utc timestamp for Highcharts

        ht = np.array(hght)

        height = ht - geoid  # Subtract the correction factor from the height
        height = np.nanmean(height) # Get the mean of the height

        return time_stamp,height #Timestamp and height for Highcharts
    else:
        return 'NULL'

def calc_jason_ts(lat1,lon1,lat2,lon2,start_date,end_date,track):

    ts_plot = []
    lon_=(float(lon1)+float(lon2))/2
    lat_=(float(lat1)+float(lat2))/2

    corr = geoidalCorrection(lon_,lat_)

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
                            time_stamp,hgt = parse_netCDF(os.path.join(working_dir,file),[lat1,lat2],corr)
                        except Exception: # If it returns NULL, move on to the next file.
                            continue
                        ts_plot.append([time_stamp,round(float(hgt),3)]) # Return this to the frontend

    return ts_plot

# calc_jason_ts('6.567022920438291','-0.22796630859374875','6.70343214627151','-0.27740478515624895','2008-01-01','2008-12-01','046')
# calc_jason_ts('12.002012','105.481','12.019098','105.481','2008-01-01','2008-12-01','140')
# calc_jason_ts('12.492662','104.483','12.533555','104.483','2008-01-01','2008-12-01','001')