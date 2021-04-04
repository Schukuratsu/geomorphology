from matplotlib import pyplot as plt
# from scipy import ndimage
from .DEM import DEM
import numpy as np
import cv2
import pandas as pd


# def gaussian_filter(arr, sigma=3):
#     return ndimage.gaussian_filter(arr, sigma=sigma)


def laplacianCv2(mat):
    filteredMat = cv2.Laplacian(mat, cv2.CV_16S, ksize=3)
    return filteredMat


def laplacian4(mat):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filteredMat = cv2.filter2D(mat, cv2.CV_16S, kernel)
    return filteredMat


def laplacian8(mat):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filteredMat = cv2.filter2D(mat, cv2.CV_16S, kernel)
    return filteredMat


def plotHist(arr, bins=None, limits=None, show=True):
    if not limits:
        # avoid the -1 on finding the min value
        maxValue = np.max(arr)
        tempArr = np.copy(arr)
        tempArr[tempArr == -1] = maxValue
        minValue = np.min(tempArr)
        limits = [minValue, maxValue]
    if not bins:
        bins = limits[1] - limits[0]
    hist = plt.hist(arr.ravel(), bins, limits)
    if show:
        plt.show()
    return hist


def showImg(mat, console=True):
    if console:
        print(mat)
    absMat = cv2.convertScaleAbs(mat)
    cv2.imshow("test", absMat)
    cv2.waitKey(0)


def getSimpleRadius(mat, center, nullValue=-1):
    yi, xi = center
    for x in range(xi, -1, -1):
        if mat[yi, x] == nullValue:
            return xi - x


def main():

    print('initiating histogram extractor.')

    demFileName = "./assets/dems/S028-030_W053-055.tif"

    optionsArray = [
         {
            #  ARENITO
            "segmentedFileName": "./assets/maps/Arenito recortado (completo).tif",
            "seedsFileName": "./assets/seeds/Centroides Arenito (completo).csv",
            # # raw
            # "outputHistogramFileName": "./assets/histograms/Arenito(raw).csv",
            # "limits": [0,1000],
            # # laplacian4
            # "outputHistogramFileName": "./assets/histograms/Arenito(laplacian4).csv",
            # "heightMapFilter": laplacian4,
            # "limits": [-500,500],
            # laplacian8
            "outputHistogramFileName": "./assets/histograms/Arenito(laplacian8).csv",
            "heightMapFilter": laplacian8,
            "limits": [-500,500],
        },
        # { 
        #     # BASALTO
        #     "segmentedFileName": "./assets/maps/Basalto recortado.tif",
        #     "seedsFileName": "./assets/seeds/Centroides Basalto.csv",
        #     # # raw
        #     # "outputHistogramFileName": "./assets/histograms/Basalto(raw).csv",
        #     # "limits": [0,1000],
        #     # laplacian4
        #     # "outputHistogramFileName": "./assets/histograms/Basalto(laplacian4).csv",
        #     # "heightMapFilter": laplacian4,
        #     # "limits": [-500,500],
        #     # laplacian8
        #     "outputHistogramFileName": "./assets/histograms/Basalto(laplacian8).csv",
        #     "heightMapFilter": laplacian8,
        #     "limits": [-500,500],
        # },
    ]

    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()


    for options in optionsArray:

        print('reading segmented DEM file "{}"...'.format(options["segmentedFileName"]))
        segmentedMap = DEM(options["segmentedFileName"])
        segmentedHeightMap = segmentedMap.getNumpyHeightMap()

        print('reading seeds file "{}"...'.format(options["seedsFileName"]))
        seeds = pd.read_csv(options["seedsFileName"], ",")

        length = seeds.values.shape[0]

        segmentedHistograms = np.empty((options['limits'][1]-options['limits'][0], length))

        print('generating histograms...')
        for seedIndex in range(length):
            seedName = seeds.values[seedIndex, 1]
            seedDegreesX = seeds.values[seedIndex, 3]
            seedDegreesY = seeds.values[seedIndex, 4]
            seedX, seedY = segmentedMap.convertDegreesToPixels(
                seedDegreesX, seedDegreesY
            )

            radius = getSimpleRadius(segmentedHeightMap, (seedY, seedX))
            # # SIMPLE INNER SQUARE
            # radius = round(np.floor(radius* np.sin(np.pi / 5)- 1))

            segmentedArea = demHeightMap[
                seedY - radius : seedY + radius, seedX - radius : seedX + radius
            ]

            if "heightMapFilter" in options:
                segmentedArea = options["heightMapFilter"](segmentedArea)

            # print(seedName, seedY, seedX, radius)
            # print(np.max(segmentedArea), np.min(segmentedArea))
            # showImg(segmentedArea, console=True)

            n, bins, patches = plotHist(segmentedArea, limits=options['limits'], show=False)

            segmentedHistograms[:, seedIndex] = np.array(n)

        print('saving histograms to file "{}"...'.format(options["outputHistogramFileName"]))
        histogramsDF = pd.DataFrame(segmentedHistograms)
        histogramsDF.to_csv(
            options["outputHistogramFileName"], header=seeds.values[:, 1]
        )

        if segmentedMap:
            segmentedMap.clear()

    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
