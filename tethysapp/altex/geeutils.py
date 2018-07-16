import ee

try:
    ee.Initialize()
except EEException as e:
    from oauth2client.service_account import ServiceAccountCredentials
    credentials = ServiceAccountCredentials.from_p12_keyfile(
    service_account_email='',
    filename='',
    private_key_password='notasecret',
    scopes=ee.oauth.SCOPE + ' https://www.googleapis.com/auth/drive ')
    ee.Initialize(credentials)

geom = ee.Geometry.Rectangle([-180,-90,180,90])


def historicalMap(algorithm='JRC',aoi=geom):

    if algorithm in ['JRC']:
        jrc = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')
        occurrence = jrc.select('occurrence')
        water = occurrence.clip(aoi).rename('water').updateMask(occurrence.gt(15))
        waterMap = water.visualize(min=0,max=100,bands='water',palette='#e8e8e8,#00008b').getMapId()

    else:
        raise NotImplementedError('Selected algorithm not available. Options are: "JRC"')

    return waterMap
