# poetry run histogram -o 0 ; poetry run histogram -o 1 ; poetry run histogram -o 2 ; poetry run histogram -o 3 ; poetry run histogram -o 4 ; poetry run histogram -o 5

from matplotlib import pyplot as plt
import getopt
import sys
# from scipy import ndimage
from .DEM import DEM
import numpy as np
import cv2
import pandas as pd

### argument handler
option = ""

errorMessage = "poetry run histogram -o <optionIndex>"
try:
    opts, args = getopt.getopt(sys.argv[1:], "ho:", ["option="])
except getopt.GetoptError:
    print(errorMessage)
    sys.exit(2)
for opt, arg in opts:
    if opt == "-h":
        print(errorMessage)
        sys.exit()
    elif opt in ("-o", "--option"):
        option = int(arg)


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
    hist, bins, patches = plt.hist(arr.ravel(), bins, limits)
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

    print("initiating histogram extractor.")

    demFileName = "./assets/dems/S028-030_W053-055.tif"

    optionsArray = [
        # 0 - arenito raw
        {
            "segmentedFileName": "./assets/maps/Arenito recortado (completo).tif",
            "seedsFileName": "./assets/seeds/Centroides Arenito (completo).csv",
            "outputHistogramFileName": "./assets/histograms/Arenito(raw).csv",
            "limits": [0, 1000],
        },
        # 1 - arenito laplacian4
        {
            "segmentedFileName": "./assets/maps/Arenito recortado (completo).tif",
            "seedsFileName": "./assets/seeds/Centroides Arenito (completo).csv",
            "outputHistogramFileName": "./assets/histograms/Arenito(laplacian4).csv",
            "heightMapFilter": laplacian4,
            "limits": [-500, 500],
        },
        # 2 - arenito laplacian8
        {
            "segmentedFileName": "./assets/maps/Arenito recortado (completo).tif",
            "seedsFileName": "./assets/seeds/Centroides Arenito (completo).csv",
            "outputHistogramFileName": "./assets/histograms/Arenito(laplacian8).csv",
            "heightMapFilter": laplacian8,
            "limits": [-500, 500],
        },
        # 3 - arenito raw
        {
            "segmentedFileName": "./assets/maps/Basalto recortado.tif",
            "seedsFileName": "./assets/seeds/Centroides Basalto.csv",
            "outputHistogramFileName": "./assets/histograms/Basalto(raw).csv",
            "limits": [0, 1000],
        },
        # 4 - arenito laplacian4
        {
            "segmentedFileName": "./assets/maps/Basalto recortado.tif",
            "seedsFileName": "./assets/seeds/Centroides Basalto.csv",
            "outputHistogramFileName": "./assets/histograms/Basalto(laplacian4).csv",
            "heightMapFilter": laplacian4,
            "limits": [-500, 500],
        },
        # 5 - arenito laplacian8
        {
            "segmentedFileName": "./assets/maps/Basalto recortado.tif",
            "seedsFileName": "./assets/seeds/Centroides Basalto.csv",
            "outputHistogramFileName": "./assets/histograms/Basalto(laplacian8).csv",
            "heightMapFilter": laplacian8,
            "limits": [-500, 500],
        },
    ]

    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()

    for options in [optionsArray[option]]:

        print('reading segmented DEM file "{}"...'.format(options["segmentedFileName"]))
        segmentedMap = DEM(options["segmentedFileName"])
        segmentedHeightMap = segmentedMap.getNumpyHeightMap()

        print('reading seeds file "{}"...'.format(options["seedsFileName"]))
        seeds = pd.read_csv(options["seedsFileName"], ",")

        cols = seeds.values.shape[0]
        rows = options["limits"][1] - options["limits"][0]

        segmentedHistograms = np.empty((rows, cols))
        # min, max, average, std dev
        segmentedData = np.empty((4, cols))

        print("generating histograms...")
        for seedIndex in range(cols):
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

            hist = plotHist(segmentedArea, limits=options["limits"], show=False)

            histogram = np.array(hist)
            numberOfPixels = np.sum(histogram)
            estatisticalDistribution = np.divide(histogram, numberOfPixels)

            segmentedHistograms[:, seedIndex] = estatisticalDistribution

            minValue = np.min(segmentedArea)
            maxValue = np.max(segmentedArea)
            meanValue = np.mean(segmentedArea)
            stdValue = np.std(segmentedArea)

            segmentedData[:, seedIndex] = np.array(
                [minValue, maxValue, meanValue, stdValue]
            )

        # adds labels to data
        indexedData = np.hstack(
            (
                np.array([["min"], ["max"], ["mean"], ["std"]]),
                segmentedData,
            )
        )

        # adds labels to histogram values
        indexedHistograms = np.hstack(
            (
                np.reshape(
                    np.arange(options["limits"][0], options["limits"][1]), (rows, 1)
                ),
                segmentedHistograms,
            )
        )

        # merges data and histograms
        mergedData = np.vstack(
            (
                indexedData,
                indexedHistograms,
            )
        )

        # adds index header label
        header = np.concatenate((np.array(["index"]), seeds.values[:, 1].flatten()))

        histogramsDF = pd.DataFrame(mergedData)

        print(
            'saving histograms to file "{}"...'.format(
                options["outputHistogramFileName"]
            )
        )

        histogramsDF.to_csv(
            options["outputHistogramFileName"],
            header=header,
            index=False,
        )

        if segmentedMap:
            segmentedMap.clear()

    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
