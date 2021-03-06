from django.http import JsonResponse
import json
from jason import *

def api_get_timeseries(request):

    json_obj = {}

    if request.method == 'GET':

        lat1 = None
        lat2 = None
        start_date = None
        end_date = None
        track = None
        sensor = None

        if request.GET.get('lat1'):
            lat1 = request.GET['lat1']
        if request.GET.get('lat2'):
            lat2 = request.GET['lat2']
        if request.GET.get('start_date'):
            start_date = request.GET['start_date']
        if request.GET.get('end_date'):
            end_date = request.GET['end_date']
        if request.GET.get('track'):
            track = request.GET['track']
        if request.GET.get('sensor'):
            sensor = request.GET['sensor']

        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')  # .strftime('%d/%m/%Y')
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')  # .strftime('%d/%m/%Y')

        try:
            ts_plot = calc_jason_ts(lat1,lat2,start_date,end_date,track,sensor)
            json_obj["values"] = ts_plot
            json_obj["lat1"] = round(float(lat1),2)
            json_obj["lat2"] = round(float(lat2),2)
            json_obj["success"] = "success"

        except Exception as e:
            json_obj["error"] = "Error Processing Request. Error: "+ str(e)

    return JsonResponse(json_obj)