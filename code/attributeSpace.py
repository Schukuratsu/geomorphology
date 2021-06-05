from .DEM import DEM
import os
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

    # configurações
    segmentedAreaFolder = "./assets/segmentedAreas"
    # tamanho da janela
    windowSize = 5

    # inicialização do resultado
    result = []

    # itera pelos arquivos
    for folder, subfolders, files in os.walk(segmentedAreaFolder):
        for file in files:
            print('reading segmentedArea file "{}"...'.format(file))
            segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
            classification = file[:7]

            # pegar tamanho pelo nome do arquivo
            diameter = file[10:-4]
            for i in range(len(diameter) - 1, -1, -1):
                if diameter[i] == "-":
                    diameter = diameter[i + 1 :]
            diameter = int(diameter)
            print(diameter)

            if windowSize > diameter:
                continue

            # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
            for windowVerticalIndex in range(int(np.floor(diameter / windowSize))):
                temp = segmentedAreaDF.values[
                    diameter * windowVerticalIndex : diameter * windowVerticalIndex
                    + windowSize * diameter,
                    :,
                ]
                for windowHorizontalIndex in range(
                    int(np.floor(diameter / windowSize))
                ):
                    subSegmentedArea = []
                    for verticalIndex in range(windowSize):
                        subSegmentedArea.append(
                            segmentedAreaDF.values[
                                diameter * verticalIndex
                                + windowSize
                                * windowHorizontalIndex : diameter
                                * verticalIndex
                                + windowSize * windowHorizontalIndex
                                + windowSize,
                                :,
                            ]
                        )
                    step = subSegmentedArea[0]
                    for sub in subSegmentedArea[1:]:
                        step = np.vstack((step, sub))
                    subSegmentedArea = step

                    # # gera um valor para cada banda final
                    result.append(
                        [
                            # "raw_std",
                            np.std(subSegmentedArea[:, 0]),
                            # "sobelMag",
                            np.mean(subSegmentedArea[:, 4]),
                            # # "sobelMag_std",
                            # np.std(subSegmentedArea[:, 4]),
                            # "sobelAng",
                            np.mean(subSegmentedArea[:, 5]),
                            # "sobelAng_std",
                            np.std(subSegmentedArea[:, 5]),
                            # # "robertsAng",
                            # np.mean(subSegmentedArea[:, 9]),
                            # # "robertsAng_std",
                            # np.std(subSegmentedArea[:, 9]),
                            # # "SaRm4_m",
                            # np.mean(subSegmentedArea[:, 10]),
                            # # "SaRm8_m",
                            # np.mean(subSegmentedArea[:, 11]),
                            # # "SaRm16_m",
                            # np.mean(subSegmentedArea[:, 12]),
                            # # "SaRm32_m",
                            # np.mean(subSegmentedArea[:, 13]),
                            # # "SaRm64_m",
                            # np.mean(subSegmentedArea[:, 14]),
                            # # "SaRm128_m",
                            # np.mean(subSegmentedArea[:, 15]),
                            # "class",
                            classification,
                        ]
                    )

    # gera imagem da matriz de resultado
    # header = [a for a in segmentedAreaDF.columns.values]
    # header.append("class")
    header = np.array(
        [
            "raw_std",
            "sobelMag",
            # "sobelMag_std",
            "sobelAng",
            "sobelAng_std",
            # "robertsAng",
            # "robertsAng_std",
            # "SaRm4_m",
            # "SaRm8_m",
            # "SaRm16_m",
            # "SaRm32_m",
            # "SaRm64_m",
            # "SaRm128_m",
            "class",
        ]
    )
    resultDF = pd.DataFrame(result, None, header)
    print(resultDF)
    g = sns.PairGrid(resultDF, hue="class")
    g.map_upper(sns.histplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.histplot, kde=True)
    g.add_legend()
    # sns.pairplot(resultDF, hue="class", kind="kde", fill=True)
    plt.show()


if __name__ == "__main__":
    main()
