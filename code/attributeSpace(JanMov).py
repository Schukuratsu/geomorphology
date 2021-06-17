from .DEM import DEM
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def main():

    # configurações
    segmentedAreaFolder = "./assets/segmentedAreas"
    # tamanho da janela
    windowSize = 100

    # inicialização do resultado
    result = []

    # itera pelos arquivos
    arenitoCounter = 1
    basaltoCounter = 1
    for folder, subfolders, files in os.walk(segmentedAreaFolder):
        for file in files:
            
            segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
            classification = file[:7]

            # # for test only
            # if classification == "arenito" and arenitoCounter == 0:
            #     continue
            # if classification == "basalto" and basaltoCounter == 0:
            #     break
            

            # pegar tamanho pelo nome do arquivo
            diameter = file[10:-4]
            for i in range(len(diameter) - 1, -1, -1):
                if diameter[i] == "-":
                    diameter = diameter[i + 1 :]
            diameter = int(diameter)

            if windowSize > diameter:
                continue
            # # for test only
            # if diameter > windowSize + 30 or diameter < windowSize + 20:
            #     continue
            # if classification == "arenito":
            #     arenitoCounter-=1
            # else:
            #     basaltoCounter-=1

            print('reading segmentedArea file "{}"...'.format(file))
            print(diameter)
            # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
            for windowVerticalIndex in range(int(np.floor(diameter + 1 - windowSize))):
                temp = segmentedAreaDF.values[
                    diameter * windowVerticalIndex : diameter * windowVerticalIndex
                    + windowSize * diameter,
                    :,
                ]
                for windowHorizontalIndex in range(
                    int(np.floor(diameter + 1 - windowSize))
                ):
                    subSegmentedArea = []
                    for verticalIndex in range(windowSize):
                        subSegmentedArea.append(
                            temp[
                                diameter * verticalIndex
                                + windowHorizontalIndex : diameter
                                * verticalIndex
                                + windowHorizontalIndex
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
                            # "sobelMag_std",
                            np.std(subSegmentedArea[:, 4]),
                            # "sobelAng",
                            np.mean(subSegmentedArea[:, 5]),
                            # "sobelAng_std",
                            np.std(subSegmentedArea[:, 5]),
                            # "robertsAng",
                            np.mean(subSegmentedArea[:, 9]),
                            # "robertsAng_std",
                            np.std(subSegmentedArea[:, 9]),
                            # "SaRm4_m",
                            np.mean(subSegmentedArea[:, 10]),
                            # "SaRm8_m",
                            np.mean(subSegmentedArea[:, 11]),
                            # "SaRm16_m",
                            np.mean(subSegmentedArea[:, 12]),
                            # "SaRm32_m",
                            np.mean(subSegmentedArea[:, 13]),
                            # "SaRm64_m",
                            np.mean(subSegmentedArea[:, 14]),
                            # "SaRm128_m",
                            np.mean(subSegmentedArea[:, 15]),
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
            "sobelMag_std",
            "sobelAng",
            "sobelAng_std",
            "robertsAng",
            "robertsAng_std",
            "SaRm4_m",
            "SaRm8_m",
            "SaRm16_m",
            "SaRm32_m",
            "SaRm64_m",
            "SaRm128_m",
            "class",
        ]
    )
    resultDF = pd.DataFrame(result, None, header)
    print(resultDF)
    # g = sns.PairGrid(resultDF, hue="class")
    # g.map_upper(sns.histplot)
    # g.map_lower(sns.kdeplot)
    # g.map_diag(sns.histplot, kde=True)
    # g.add_legend()
    for label in header[:-1]:
        sns.displot(resultDF, x=label, kde=True, hue="class")
        plt.show()


if __name__ == "__main__":
    main()
