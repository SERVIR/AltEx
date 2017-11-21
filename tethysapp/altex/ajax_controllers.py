import json
from django.http import JsonResponse
import datetime
from jason import calc_jason_ts

def timeseries(request):

    return_obj = {}

    if request.is_ajax() and request.method == 'POST':

        info = request.POST
        lat1 = info.get('lat1')
        lon1 = info.get('lon1')
        lat2 = info.get('lat2')
        lon2 = info.get('lon2')
        start_date = info.get('start_date')
        end_date = info.get('end_date')
        sensor = info.get('sensor')
        track = info.get('track')
        track = track.split(' ')[2]
        sensor_name = sensor.split('|')[1]
        start_date = datetime.datetime.strptime(start_date, '%B %d, %Y') #.strftime('%d/%m/%Y')
        end_date = datetime.datetime.strptime(end_date, '%B %d, %Y')  #.strftime('%d/%m/%Y')

        if sensor_name == 'jason':
            try:
                ts_plot = calc_jason_ts(lat1,lon1,lat2,lon2,start_date,end_date,track)
                max_height = max(ts_plot, key=lambda x: x[1])
                min_height = min(ts_plot, key=lambda x: x[1])

                return_obj["max_ht"] = max_height[1]
                return_obj["min_ht"] = min_height[1]
                return_obj["values"] = ts_plot
                return_obj["lat1"] = round(float(lat1),2)
                return_obj["lat2"] = round(float(lat2),2)
                return_obj["success"] = "success"

            except Exception as e:
                return_obj["error"] = "Error Processing Request. Error: "+ str(e)

    return JsonResponse(return_obj)