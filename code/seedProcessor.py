from .DEM import DEM
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def raw(mat):
    return mat


def sobel(mat):
    Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    Gy = np.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]])
    SobelX = cv2.filter2D(mat, cv2.CV_16S, Gx)
    SobelY = cv2.filter2D(mat, cv2.CV_16S, Gy)
    return np.sqrt(np.add(np.square(SobelX), np.square(SobelY)))


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


def showImg(mat, console=True):
    if console:
        print(mat)
    absMat = cv2.convertScaleAbs(mat)
    cv2.imshow("test", absMat)
    cv2.waitKey(0)


def main():

    # arquivos de entrada
    demFileName = "./assets/dems/S028-030_W053-055.tif"
    seedsFileName = "./assets/seeds/Centroides.csv"

    # configurações
    filters = [raw, laplacianCv2, laplacian4, laplacian8, sobel]
    filterNames = ["raw", "laplacianCv2", "laplacian4", "laplacian8", "sobel"]
    seedsIndex = []

    # lendo arquivos
    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()
    seeds = pd.read_csv(seedsFileName)

    if len(seedsIndex) == 0:
        seedsIndex = range(seeds.values.shape[0])
    # iterando pelas seeds selecionadas
    for seedIndex in seedsIndex:
        seed = {
            "index": int(seeds.values[seedIndex, 0]),
            "Y": int(seeds.values[seedIndex, 4]),
            "X": int(seeds.values[seedIndex, 3]),
            "radius_px": int(seeds.values[seedIndex, 5]),
            "class": seeds.values[seedIndex, 6],
        }
        print("seed: ", seed)

        # inicializa matriz de resultado
        result = []

        # generate filename
        filename = "./assets/segmentedAreas/{}-{}-{}.csv".format(
            seed["class"], seed["index"], 2 * seed["radius_px"]
        )
        print("filename: ", filename)

        # segmenta area da seed
        seedArea = demHeightMap[
            seed["Y"] - seed["radius_px"] : seed["Y"] + seed["radius_px"],
            seed["X"] - seed["radius_px"] : seed["X"] + seed["radius_px"],
        ]

        # itera pelos filtros
        for filter in filters:

            # inicializa dataframe
            filteredArea = filter(seedArea)
            # showImg(filteredArea)

            # transforma area em array unidimensional
            result.append(filteredArea.ravel())

        # monta dataframe
        resultDF = pd.DataFrame(np.array(result).T)

        # salva arquivo
        resultDF.to_csv(filename, header=filterNames, index=False)

    # limpa cache
    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
