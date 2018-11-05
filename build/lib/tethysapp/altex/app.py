from tethys_sdk.base import TethysAppBase, url_map_maker


class Altex(TethysAppBase):
    """
    Tethys app class for AlTex.
    """

    name = 'AltEx'
    index = 'altex:home'
    icon = 'altex/images/logo.png'
    package = 'altex'
    root_url = 'altex'
    color = '#2980b9'
    description = 'Altimetry Explorer'
    tags = 'Remote Sensing'
    enable_feedback = False
    feedback_emails = []

    def url_maps(self):
        """
        Add controllers
        """
        UrlMap = url_map_maker(self.root_url)

        url_maps = (
            UrlMap(
                name='home',
                url='altex',
                controller='altex.controllers.home'
            ),
            UrlMap(
                name='timeseries',
                url='altex/timeseries',
                controller='altex.ajax_controllers.timeseries'
            ),
            UrlMap(
                name='api_get_timeseries',
                url='altex/api/timeseries',
                controller='altex.api.api_get_timeseries'
            ),
        )

        return url_maps
