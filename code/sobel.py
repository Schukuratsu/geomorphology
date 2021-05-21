from .DEM import DEM
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def sobelMagImg(mat):
    Gx = np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]])
    Gy = np.array([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]])
    SobelX = cv2.filter2D(mat, cv2.CV_64F, Gx)
    SobelY = cv2.filter2D(mat, cv2.CV_64F, Gy)
    SobelMagResult = np.sqrt(
        np.add(np.square(SobelX, dtype=np.float64), np.square(SobelY, dtype=np.float64))
    )
    AdjustedSobelMagResult = np.array(
        np.floor(
            np.add(
                np.multiply(
                    SobelMagResult,
                    (255 / 2)
                    / np.max([np.max(SobelMagResult), -np.min(SobelMagResult)]),
                ),
                255 / 2,
            ),
        ),
        dtype=np.uint8,
    )
    return AdjustedSobelMagResult


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


def showImg(mat, console=True):
    if console:
        print(mat)
    absMat = cv2.convertScaleAbs(mat)
    cv2.imshow("test", absMat)
    cv2.waitKey(0)


def main():

    # arquivos de entrada
    demFileName = "./assets/dems/S028-030_W053-055.tif"

    # lendo arquivos
    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()

    # processamento
    print("processing images")
    sobelMag = sobelMagImg(demHeightMap)
    showImg(sobelMag)
    cv2.imwrite("./resultados/imagens/sobelMag.tiff", sobelMag)
    sobelAng = sobelAngImg(demHeightMap)
    showImg(sobelAng)
    cv2.imwrite("./resultados/imagens/sobelAng.tiff", sobelAng)

    # limpa cache
    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
