from matplotlib import pyplot as plt

# from scipy import ndimage
from .DEM import DEM
import numpy as np
import cv2
import pandas as pd


LIMITS = [0, 1000]


def gaussian_filter(arr, sigma=3):
    return ndimage.gaussian_filter(arr, sigma=sigma)


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


def showImg(arr):
    cv2.imshow("test", arr)
    cv2.waitKey(0)


def getSimpleRadius(mat, center, nullValue=-1):
    yi, xi = center
    for x in range(xi, -1, -1):
        if mat[yi, x] == nullValue:
            return xi - x


def main():

    global LIMITS

    arenitoSegmentedFileName = "./assets/maps/Arenito recortado (completo).tif"
    arenitoSeedsFileName = "./assets/seeds/Centroides Arenito (completo).csv"
    arenitoOutputHistogramFileName = "./assets/histograms/Arenito (completo).csv"
    basaltoSegmentedFileName = "./assets/maps/Basalto recortado.tif"
    basaltoSeedsFileName = "./assets/seeds/Centroides Basalto.csv"
    basaltoOutputHistogramFileName = "./assets/histograms/Basalto.csv"
    demFileName = "./assets/dems/S028-030_W053-055.tif"

    fileNamesArray = [
        # (
        #     arenitoSegmentedFileName,
        #     arenitoSeedsFileName,
        #     arenitoOutputHistogramFileName,
        # ),
        (basaltoSegmentedFileName, basaltoSeedsFileName, basaltoOutputHistogramFileName)
    ]

    # dem = DEM(demFileName)
    # demHeightMap = dem.getNumpyHeightMap()

    for segmentedFileName, seedsFileName, outputHistogramFileName in fileNamesArray:
        segmentedMap = DEM(segmentedFileName)
        segmentedHeightMap = segmentedMap.getNumpyHeightMap()
        seeds = pd.read_csv(seedsFileName, ",")

        length = seeds.values.shape[0]

        segmentedHistograms = np.empty((LIMITS[1], length))

        for seedIndex in range(length):
            seedName = seeds.values[seedIndex, 1]
            seedDegreesX = seeds.values[seedIndex, 3]
            seedDegreesY = seeds.values[seedIndex, 4]
            seedX, seedY = segmentedMap.convertDegreesToPixels(
                seedDegreesX, seedDegreesY
            )

            radius = getSimpleRadius(segmentedHeightMap, (seedY, seedX))
            # # SIMPLE INNER SQUARE
            # radius = round(
            #     np.floor(
            #         getSimpleRadius(segmentedHeightMap, (seedY, seedX))
            #         * np.sin(np.pi / 5)
            #         - 1
            #     )
            # )

            segmentedArea = segmentedHeightMap[
                seedY - radius : seedY + radius, seedX - radius : seedX + radius
            ]

            # print(seedName, seedY, seedX, radius)
            # print(np.max(segmentedArea), np.min(segmentedArea))
            # showImg(segmentedArea)

            n, bins, patches = plotHist(segmentedArea, limits=LIMITS, show=False)

            segmentedHistograms[:, seedIndex] = np.array(n)

        histogramsDF = pd.DataFrame(segmentedHistograms)
        histogramsDF.to_csv(outputHistogramFileName, header=seeds.values[:, 1])

        segmentedMap.clear()

    # dem.clear()

    # # plotHist(arr1)
    # arr2=cv2.threshold(arr1,150,255,cv2.THRESH_BINARY)
    # arr3=np.mat(arr2,dtype=np.uint8)
    # showImg(arr3)

    # arr2 = cv2.normalize(arr1, 0, 255, cv2.NORM_MINMAX)
    # plt.hist(arr2.ravel(),256,[0,255]); plt.show()
    # print(arr2[3000:-3000,3000:-3000])
    # arr3 = cv2.resize(arr2, (800, 800))
    # print(arr3[350:-350,350:-350])
    # arr4 = np.array(arr3, dtype=np.uint8)
    # print(arr4[350:-350,350:-350])


if __name__ == "__main__":
    main()
