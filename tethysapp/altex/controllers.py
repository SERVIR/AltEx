from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from tethys_sdk.gizmos import *
from jason import geoidalCorrection

def home(request):
    """
    Controller for the app home page.
    """
    start_date = DatePicker(name='start-date',
                            attributes={'id': 'start-date'},
                            display_text='Start Date',
                            autoclose=True,
                            format='MM d, yyyy',
                            start_date='1/1/2008',
                            start_view='decade',
                            today_button=True,
                            initial='January 1, 2008')

    end_date = DatePicker(name='end-date',
                          attributes={'id': 'end-date'},
                          display_text='End Date',
                          autoclose=True,
                          format='MM d, yyyy',
                          start_date='1/1/2008',
                          start_view='decade',
                          today_button=True,
                          initial='December 1, 2008')

    select_sat = SelectInput(display_text='Select Sensor',
                             name='select-sat',
                             attributes={'id':'select-sat'},
                             multiple=False,
                             options=[('JASON 2', '1|jason')])
    # , ('SARAL', '2|saral')

    context = {
        'start_date':start_date,
        'end_date':end_date,
        'select_sat':select_sat

    }

    return render(request, 'altex/home.html', context)