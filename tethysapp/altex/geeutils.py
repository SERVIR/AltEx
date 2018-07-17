import ee
import math

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


def extractS3Water(img):
    qa = img.select('quality_flags');
    waterBit = int(math.pow(2,31))
    waterFlag = qa.bitwiseAnd(waterBit).eq(0)
    return waterFlag.rename('water')


def historicalMap(algorithm='JRC',aoi=geom):

    ic = ee.ImageCollection('COPERNICUS/S3/OLCI').filterDate('2017-03-01','2017-03-31')

    icWater = ic.map(extractS3Water)

    oceanObs = icWater.sum()
    allObs = icWater.count()

    ocean = oceanObs.divide(allObs).multiply(100)

    if algorithm in ['JRC']:
        jrc = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')
        occurrence = jrc.select('occurrence').unmask().rename('water')
        oceanInland = occurrence.add(ocean)
        water = oceanInland.updateMask(oceanInland.gt(15))
        waterMap = water.visualize(min=0,max=100,bands='water',palette='#e8e8e8,#00008b').getMapId()

    else:
        raise NotImplementedError('Selected algorithm not available. Options are: "JRC"')

    return waterMap
