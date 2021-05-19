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


def main():

    # inicialização

    demFileName = "./assets/dems/S028-030_W053-055.tif"
    seedsFileName = "./assets/seeds/Centroides.csv"

    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()

    # seleciona a seed
    seeds = pd.read_csv(seedsFileName)
    seedIndex = 31
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

    # inicializa dataframe
    result = pd.DataFrame()

    # itera sobre os pixels de ambas as bandas ao mesmo tempo
    print("gerando espaço de atributos")
    for y in range(band1.shape[0]):
        for x in range(band1.shape[1]):

            # incrementa elemento da matriz de resultado baseado no valor das bandas naquele pixel
            result = result.append([[bands[0][y, x], bands[1][y, x]]], True)

    # gera imagem da matriz de resultado
    print(result)
    sns.displot(result, x=0, y=1, binwidth=(10,1))
    plt.xlim((300,800))
    plt.ylim((-20,20))
    plt.show()

    # limpa cache
    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
