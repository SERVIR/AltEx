/*****************************************************************************
 * FILE:    MAP JS
 * DATE:    30 October 2017
 * AUTHOR: Sarva Pulla
 * COPYRIGHT: (c) SERVIR GLOBAL 2017
 * LICENSE: BSD 2-Clause
 *****************************************************************************/

/*****************************************************************************
 *                      LIBRARY WRAPPER
 *****************************************************************************/

var LIBRARY_OBJECT = (function() {
    // Wrap the library in a package function
    "use strict"; // And enable strict mode for this library

    /************************************************************************
     *                      MODULE LEVEL / GLOBAL VARIABLES
     *************************************************************************/
    var current_layer,
        gs_wms_workspace,
        gs_wms_url,
        jason2_store,
        saral_store,
        layers,
        $loading,
        map,
        public_interface,				// Object returned by the module
        variable_data,
        water_mapid,
        water_token;
    /************************************************************************
     *                    PRIVATE FUNCTION DECLARATIONS
     *************************************************************************/

    var init_all,
        init_events,
        init_vars,
        init_map;

    /************************************************************************
     *                    PRIVATE FUNCTION IMPLEMENTATIONS
     *************************************************************************/

    init_vars = function(){
        water_mapid = $('#layers').attr('data-water-mapid');
        water_token = $('#layers').attr('data-water-token');
        var $layers_element = $('#layers');
        $loading = $('#view-file-loading');
        var $var_element = $("#variable");
        // variable_data = $var_element.attr('data-variable-info');
        // variable_data = JSON.parse(variable_data);
        gs_wms_url = $var_element.attr('data-geoserver-url');
        // wms_url = JSON.parse(wms_url);
        gs_wms_workspace = $var_element.attr('data-geoserver-workspace');
        jason2_store = $var_element.attr('data-jason2-store');
        saral_store = $var_element.attr('data-saral-store');

    };

    init_map = function(){
        var attribution = new ol.Attribution({
            html: 'Tiles © <a href="https://services.arcgisonline.com/ArcGIS/rest/services/">ArcGIS</a>'
        });

        var base_map = new ol.layer.Tile({
            crossOrigin: 'anonymous',
            source: new ol.source.XYZ({
                attributions: [attribution],
                url: 'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/' +
                'World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'
            })
        });

        var water_layer = new ol.layer.Tile({
            source: new ol.source.XYZ({
                url: "https://earthengine.googleapis.com/map/"+water_mapid+"/{z}/{x}/{y}?token="+water_token
            })
        });

        var jason2_style = new ol.style.Style({
            fill: new ol.style.Fill({
                color: 'red'
            }),
            stroke: new ol.style.Stroke({
                color: 'red',
                width: 4
            })
        });

        var jason2_layer = new ol.layer.Image({
            source: new ol.source.ImageWMS({
                // url: 'http://tethys.servirglobal.net:8181/geoserver/wms',
                url: gs_wms_url,
                params: {'LAYERS':gs_wms_workspace+':'+jason2_store},
                serverType: 'geoserver',
                crossOrigin: 'Anonymous'
            })
        });

        var saral_style = new ol.style.Style({
            fill: new ol.style.Fill({
                color: 'blue'
            }),
            stroke: new ol.style.Stroke({
                color: 'blue',
                width: 4
            })
        });

        var saral_layer = new ol.layer.Image({
            source: new ol.source.ImageWMS({
                url: gs_wms_url,
                params: {'LAYERS':gs_wms_workspace+':'+jason2_store},
                serverType: 'geoserver',
                crossOrigin: 'Anonymous'
            })
        });

        // var base_map = new ol.layer.Tile({
        //     source: new ol.source.BingMaps({
        //         key: '5TC0yID7CYaqv3nVQLKe~xWVt4aXWMJq2Ed72cO4xsA~ApdeyQwHyH_btMjQS1NJ7OHKY8BK-W-EMQMrIavoQUMYXeZIQOUURnKGBOC7UCt4',
        //         imagerySet: 'AerialWithLabels' // Options 'Aerial', 'AerialWithLabels', 'Road'
        //     })
        // });

        var pt1_source = new ol.source.Vector();
        var pt1_vector = new ol.layer.Vector({
            source: pt1_source,
            style: new ol.style.Style({
                fill: new ol.style.Fill({
                    color: 'rgba(255, 255, 255, 0.2)'
                }),
                stroke: new ol.style.Stroke({
                    color: '#ffcc33',
                    width: 2
                }),
                image: new ol.style.Circle({
                    radius: 7,
                    fill: new ol.style.Fill({
                        color: '#ffcc33'
                    })
                })
            })
        });

        var pt2_source = new ol.source.Vector();
        var pt2_vector = new ol.layer.Vector({
            source: pt2_source,
            style: new ol.style.Style({
                fill: new ol.style.Fill({
                    color: 'green'
                }),
                stroke: new ol.style.Stroke({
                    color: 'green',
                    width: 2
                }),
                image: new ol.style.Circle({
                    radius: 7,
                    fill: new ol.style.Fill({
                        color: 'green'
                    })
                })
            })
        });

        //Note: The Satellite Tracks layers need to be in the same order as the dropdown as generated in the controllers.py

        layers = [base_map,water_layer,jason2_layer,saral_layer,pt1_vector,pt2_vector];

        map = new ol.Map({
            target: 'map',
            view: new ol.View({
                center: ol.proj.transform([0, 18.6], 'EPSG:4326', 'EPSG:3857'),
                zoom: 4,
                minZoom: 2,
                maxZoom: 18
            }),
            layers: layers
        });


        var modify = new ol.interaction.Modify({source: pt1_source});
        var modify2 = new ol.interaction.Modify({source: pt2_source});

        map.addInteraction(modify);
        map.addInteraction(modify2);

        var draw, snap; // global so we can remove them later

        function addInteractions() {
            draw = new ol.interaction.Draw({
                source: pt1_source,
                type: 'Point'
            });
            map.addInteraction(draw);
            draw.on('drawstart', function (e) {
                pt1_source.clear();
            });

            draw.on('drawend', function (e) {
                map.removeInteraction(draw);
                map.removeInteraction(snap);
                $("#btn-pt2").click();
            });

            snap = new ol.interaction.Snap({source: pt1_source});
            map.addInteraction(snap);

        }

        function addInteractions2() {
            draw = new ol.interaction.Draw({
                source: pt2_source,
                type: 'Point'
            });
            map.addInteraction(draw);
            draw.on('drawstart', function (e) {
                pt2_source.clear();
            });

            draw.on('drawend', function (e) {
                map.removeInteraction(draw);
                map.removeInteraction(snap);
            });

            snap = new ol.interaction.Snap({source: pt2_source});
            map.addInteraction(snap);

        }

        $('#btn-pt1').on('click', function(){
            addInteractions();
        });

        $('#btn-pt2').on('click', function(){
            addInteractions2();
        });

        pt1_vector.getSource().on('addfeature', function(event){
            var feature_json = saveData(pt1_vector);
            var parsed_feature = JSON.parse(feature_json);
            var coords = parsed_feature["features"][0]["geometry"]["coordinates"];
            var proj_coords = ol.proj.transform(coords, 'EPSG:3857','EPSG:4326');
            getTrack(coords,'pt1');
            $("#point1").val(proj_coords);
        });

        pt2_vector.getSource().on('addfeature', function(event){
            var feature_json = saveData(pt2_vector);
            var parsed_feature = JSON.parse(feature_json);
            var coords = parsed_feature["features"][0]["geometry"]["coordinates"];
            var proj_coords = ol.proj.transform(coords, 'EPSG:3857','EPSG:4326');
            getTrack(coords,'pt2');
            $("#point2").val(proj_coords);
        });

        pt1_vector.getSource().on('changefeature', function(event){
            var feature_json = saveData(pt1_vector);
            var parsed_feature = JSON.parse(feature_json);
            var coords = parsed_feature["features"][0]["geometry"]["coordinates"];
            var proj_coords = ol.proj.transform(coords, 'EPSG:3857','EPSG:4326');
            getTrack(coords,'pt1');
            $("#point1").val(proj_coords);
        });

        pt2_vector.getSource().on('changefeature', function(event){
            var feature_json = saveData(pt2_vector);
            var parsed_feature = JSON.parse(feature_json);
            var coords = parsed_feature["features"][0]["geometry"]["coordinates"];
            var proj_coords = ol.proj.transform(coords, 'EPSG:3857','EPSG:4326');
            getTrack(coords,'pt2');
            $("#point2").val(proj_coords);
        });

        //Save the drawn feature as a json object
        function saveData(layer) {
            // get the format the user has chosen
            var data_type = 'GeoJSON',
                // define a format the data shall be converted to
                format = new ol.format[data_type](),
                // this will be the data in the chosen format
                data;
            try {
                // convert the data of the vector_layer into the chosen format
                data = format.writeFeatures(layer.getSource().getFeatures());
            } catch (e) {
                console.log(e);
                return;
            }
            return data;
        }

        function getTrack(coords,pt){
            var view = map.getView();
            var viewResolution = view.getResolution();

            var sensor = $("#select-sat").val();

            var index = sensor.split('|')[0] + 1;

            current_layer = layers[index];

            var wms_url = current_layer.getSource().getGetFeatureInfoUrl(coords, viewResolution, view.getProjection(), {'INFO_FORMAT': 'application/json'});

            if (wms_url) {

                $.ajax({
                    type: "GET",
                    url: wms_url,
                    dataType: 'json',
                    success: function (result) {
                        var name = result["features"][0]["properties"]["Name"];
                        if(pt == 'pt1'){
                            $("#track1").val(name);
                        }else{
                            $("#track2").val(name);
                        }
                    },
                    error: function (XMLHttpRequest, textStatus, errorThrown) {
                        console.warn('Not on a track');
                    }
                });
            }
        }


    };

    init_events = function() {
        (function () {
            var target, observer, config;
            // select the target node
            target = $('#app-content-wrapper')[0];

            observer = new MutationObserver(function () {
                window.setTimeout(function () {
                    map.updateSize();
                }, 350);
            });
            $(window).on('resize', function () {
                map.updateSize();
            });

            config = {attributes: true};

            observer.observe(target, config);
        }());

        map.on("moveend", function() {
            var zoom = map.getView().getZoom();
        });

    };

    init_all = function(){
        init_vars();
        init_map();
        init_events();
    };


    /************************************************************************
     *                        DEFINE PUBLIC INTERFACE
     *************************************************************************/
    /*
     * Library object that contains public facing functions of the package.
     * This is the object that is returned by the library wrapper function.
     * See below.
     * NOTE: The functions in the public interface have access to the private
     * functions of the library because of JavaScript function scope.
     */
    public_interface = {

    };

    /************************************************************************
     *                  INITIALIZATION / CONSTRUCTOR
     *************************************************************************/

    // Initialization: jQuery function that gets called when
    // the DOM tree finishes loading

    $(function() {
        init_all();

        $("#btn-submit").on('click',function(){
            $('.warning').html('');
            $loading.removeClass('hidden');
            $("#plotter").addClass('hidden');
            var point1 = $("#point1").val();
            var point2 = $("#point2").val();
            var start_date = $("#start-date").val();
            var end_date = $("#end-date").val();
            var sensor = $("#select-sat").val();
            var track1 = $("#track1").val();
            var track2 = $("#track2").val();
            if (point1 == ""){
                $('.warning').html('<b>Please select a lower bound.</b>');
                $loading.addClass('hidden');
                $("#plotter").addClass('hidden');
                return false;
            }
            if (point2 == ""){
                $('.warning').html('<b>Please select an upper bound.</b>');
                $loading.addClass('hidden');
                $("#plotter").addClass('hidden');
                return false;
            }

            if (track1 != track2){
                $('.warning').html('<b>The points have to be on the same track. Please make the necessary changes and try again.</b>');
                $loading.addClass('hidden');
                $("#plotter").addClass('hidden');
                return false;
            }
            if((start_date == "") || (end_date == "")){
                $('.warning').html('<b>Please select a start/end date before submitting.</b>');
                $loading.addClass('hidden');
                $("#plotter").addClass('hidden');
                return false;
            }

            var lwr_pt = point1.split(",");
            var upr_pt = point2.split(",");
            var wgs84_sphere = new ol.Sphere(6378137);
            var distance = wgs84_sphere.haversineDistance([parseFloat(lwr_pt[0]),parseFloat(lwr_pt[1])],[parseFloat(upr_pt[0]),parseFloat(upr_pt[1])]);
            if(distance < 350){
                $('.warning').html('<b>The distance between the points has to be greater than 350 m. Please make the necessary changes and try again.</b>');
                $loading.addClass('hidden');
                $("#plotter").addClass('hidden');
                return false;
            }

            var xhr = ajax_update_database('timeseries',{'sensor':sensor,'lat1':parseFloat(lwr_pt[1]),'lon1':parseFloat(lwr_pt[0]),'lat2':parseFloat(upr_pt[1]),'lon2':parseFloat(upr_pt[0]),'start_date':start_date,'end_date':end_date,'track':track1});

            xhr.done(function(data) {
                if("success" in data) {
                    // var json_response = JSON.parse(data);
                    $loading.addClass('hidden');

                    if(data.values.length > 0){
                        $("#plotter").removeClass('hidden');
                        Highcharts.stockChart('plotter',{
                            chart: {
                                type:'spline',
                                zoomType: 'x',
                                height: 350
                            },
                            title: {
                                text:"Values at between Lat 1: " +data.lat1+' and Lat 2: '+data.lat2,
                                style: {
                                    fontSize: '14px'
                                }
                            },
                            tooltip: {
                                xDateFormat: '%Y-%m-%d'
                            },
                            // xAxis: {
                            //     type: 'datetime',
                            //     labels: {
                            //         format: '{value:%d %b %Y}'
                            //         // rotation: 90,
                            //         // align: 'left'
                            //     },
                            //     title: {
                            //         text: 'Date'
                            //     }
                            // },
                            xAxis: {
                                type: 'datetime',
                                dateTimeLabelFormats: { // don't display the dummy year
                                    month: '%e. %b',
                                    year: '%b'
                                },
                                title: {
                                    text: 'Date'
                                }
                            },
                            yAxis: {
                                min: data.min_ht,
                                max: data.max_ht,
                                title: {
                                    text: "Water Height (m)"
                                }

                            },
                            exporting: {
                                enabled: true
                            },
                            series: [{
                                data:data.values,
                                name: "Altimetry"
                            }]
                        });
                    }else{
                        $('.warning').html('<b>Sorry! There are no data values for the selected points. Please try another site.</b>');
                        $loading.addClass('hidden');
                        $("#plotter").addClass('hidden');
                    }

                }else{
                    $('.warning').html('<b>'+data.error+'</b>');
                    $loading.addClass('hidden');
                    $("#plotter").addClass('hidden');
                }
            });
        });

        $('#select-sat').change(function(){
            map.getLayers().item(1).setVisible(false);
            map.getLayers().item(2).setVisible(false);
            var selected_option = $(this).find('option:selected').val();
            map.getLayers().item(selected_option.split("|")[0]).setVisible(true);
        }).change();

    });

    return public_interface;

}()); // End of package wrapper
// NOTE: that the call operator (open-closed parenthesis) is used to invoke the library wrapper
// function immediately after being parsed.
