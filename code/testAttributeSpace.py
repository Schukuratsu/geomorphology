from .DEM import DEM
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


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

    # inicialização

    demFileName = "./assets/dems/S028-030_W053-055.tif"
    seedsFileName = "./assets/seeds/Centroides.csv"

    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()

    # seleciona a seed
    seeds = pd.read_csv(seedsFileName)
    seedIndex = 30
    seed = {
        "Y": int(seeds.values[seedIndex, 4]),
        "X": int(seeds.values[seedIndex, 3]),
        "radius_px": int(seeds.values[seedIndex, 5]),
        "class": seeds.values[seedIndex, 6],
    }
    print("seed: ", seed)

    # gera as "bandas" de entrada
    band1 = demHeightMap[
        seed["Y"] - seed["radius_px"] : seed["Y"] + seed["radius_px"],
        seed["X"] - seed["radius_px"] : seed["X"] + seed["radius_px"],
    ]
    band2 = laplacian4(band1)
    bands = [band1, band2]

    # inicializa matriz de resultado
    # banda 1 = x
    # banda 2 = y
    band1map = demHeightMap
    band2map = laplacian4(band1map)
    # xlimits = [int(np.floor(np.min(band1map))), int(np.ceil(np.max(band1map)))]
    # ylimits = [int(np.floor(np.min(band2map))), int(np.ceil(np.max(band2map)))]
    xlimits = [int(np.floor(np.min(band1))), int(np.ceil(np.max(band1)))]
    ylimits = [int(np.floor(np.min(band2))), int(np.ceil(np.max(band2)))]
    rows = ylimits[1] - ylimits[0]
    cols = xlimits[1] - xlimits[0]
    result = np.empty((rows, cols), dtype=np.uint32)
    print("xlimits: ", xlimits)
    print("ylimits: ", ylimits)
    print("rows: ", rows)
    print("cols: ", cols)

    # itera sobre os pixels de ambas as bandas ao mesmo tempo
    print("gerando espaço de atributos")
    for y in range(band1.shape[0]):
        for x in range(band1.shape[1]):

            # incrementa elemento da matriz de resultado baseado no valor das bandas naquele pixel
            resX = int(np.floor(bands[0][y, x] - xlimits[0]) - 1)
            resY = int(np.floor(bands[1][y, x] - ylimits[0]) - 1)
            # print(resY, resX)
            result[resY, resX] = int(result[resY, resX] + 1)
            # print(result[resY, resX])

    # gera imagem da matriz de resultado
    normalizedImg = np.empty_like(result, dtype=np.float32)
    normalizedImg = cv2.normalize(
        np.array(result, dtype=np.float32), normalizedImg, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    )
    resizedImg = cv2.resize(normalizedImg, (1500, 1000))
    cv2.imshow("teste", resizedImg)
    # plt.hist(result, 255)
    plt.hist(result)
    plt.show()
    cv2.waitKey(0)

    # limpa cache
    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
