from os import error
import sys
from .DEM import DEM
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def sobelAngImg(mat):
    Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    Gy = np.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]])
    SobelX = cv2.filter2D(mat, cv2.CV_16S, Gx)
    SobelY = cv2.filter2D(mat, cv2.CV_16S, Gy)
    SobelAngResult = np.arctan2(SobelY, SobelX)
    AdjustedSobelAngResult = np.array(
        np.floor(np.multiply(np.add(SobelAngResult, np.pi), 255 / (2 * np.pi))),
        dtype=np.uint8,
    )
    return AdjustedSobelAngResult


def robertsMagImg(mat, window=2):
    if window % 2 == 1:
        sys.exit("Error: window param on 'robertsMag' must be even.")
    half = int(window / 2)
    ones = np.ones((half, half))
    Gx = np.zeros((window, window))
    Gx[:half, half:] = ones
    Gx[half:, :half] = -ones
    Gy = np.zeros((window, window))
    Gy[:half, :half] = ones
    Gy[half:, half:] = -ones
    RobertsX = cv2.filter2D(mat, cv2.CV_16S, Gx)
    RobertsY = cv2.filter2D(mat, cv2.CV_16S, Gy)
    RobertsMagResult = np.sqrt(
        np.add(
            np.square(RobertsX, dtype=np.float64), np.square(RobertsY, dtype=np.float64)
        )
    )
    AdjustedRobertsMagResult = np.array(
        np.floor(
            np.add(
                np.multiply(
                    RobertsMagResult,
                    (255 / 2)
                    / np.max([np.max(RobertsMagResult), -np.min(RobertsMagResult)]),
                ),
                255 / 2,
            ),
        ),
        dtype=np.uint8,
    )
    return AdjustedRobertsMagResult


def showImg(mat, console=True):
    if console:
        print(mat)
    absMat = cv2.convertScaleAbs(mat)
    cv2.imshow("test", absMat)
    cv2.waitKey(0)


def main():

    # arquivos de entrada
    demFileName = "./assets/dems/S028-030_W053-055.tif"

    # arquivos de saída
    saveFilePrefix = "./resultados/imagens/sobelAng/sobelAng_roberts_window-"

    # lendo arquivos
    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()

    # geração da imagem de input
    print("processing sobelAng")
    sobelAng = sobelAngImg(demHeightMap)
    # showImg(sobelAng)

    # processamento com diferentes tamanhos de filtro
    results = []
    windows = [2, 4, 8, 16, 32, 64, 128]
    for i in range(len(windows)):
        print("processing roberts for window: {}".format(windows[i]))
        results.append(robertsMagImg(sobelAng, windows[i]))
        cv2.imwrite(
            "{}{}.tiff".format(saveFilePrefix, windows[i]), results[i]
        )

    # limpa cache
    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
