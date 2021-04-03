# Documentação: https://gdal.org/tutorials/raster_api_tut.html

from osgeo import gdal
import numpy as np
import sys

# import cv2
# import struct
# import getopt
# import itertools
# import pandas as pd


class DEM:
    def __init__(self, fileName: str, nullValue: int = -1):
        """Instancia o dataset do arquivo selecionado"""

        self.dataset = gdal.Open(fileName, gdal.GA_ReadOnly)
        if not self.dataset:
            sys.exit()
        self.nullValue = nullValue

    def getHeightMap(self):
        """Retorna o heightmap do arquivo."""

        return self.dataset.GetRasterBand(1)

    def getNumpyHeightMap(self):
        """Retorna o heightmap do arquivo em formato NumPY"""

        return np.array(self.getHeightMap().ReadAsArray())

    def getPixelSizeInDegrees(self):
        """Retorna a largura do pixel em graus (x, y)"""

        geotransform = self.dataset.GetGeoTransform()
        return geotransform[1], geotransform[5]

    def getStartingPositionInDegrees(self):
        """Retorna a posição inicial do mapa em graus (x, y)"""

        geotransform = self.dataset.GetGeoTransform()
        return geotransform[0], geotransform[3]

    def getMapSizeInPixels(self):
        """Retorna a largura do mapa em pixels (x, y)"""

        return self.dataset.RasterXSize, self.dataset.RasterYSize

    def getMapSizeInDegrees(self):
        """Retorna a largura do mapa em graus (x, y)"""

        return self.getPixelSizeInDegrees() * self.getMapSizeInPixels()

    def convertDegreesToPixels(self, degreesX, degreesY):
        """Retorna a largura do mapa em graus (x, y)"""

        degreesXi, degreesYi = self.getStartingPositionInDegrees()
        deltaDegreesX = degreesX - degreesXi
        deltaDegreesY = degreesY - degreesYi
        pixelSizeInDegreesX, pixelSizeInDegreesY = self.getPixelSizeInDegrees()
        pixelX = round(deltaDegreesX / pixelSizeInDegreesX)
        pixelY = round(deltaDegreesY / pixelSizeInDegreesY)
        return pixelX, pixelY

    def clear(self):
        """Limpa o dataset para desalocar a memória"""

        self.dataset = None


def main():
    fileName = "./assets/dems/S028-030_W053-055.tif"
    map = DEM(fileName)
    test = np.array(map.getHeightMap().ReadAsArray())
    print(test)
    print(test.shape)
    print(test.size)
    print(test.dtype)
    map.clear()


if __name__ == "__main__":
    main()
