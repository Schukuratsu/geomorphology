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
    classesFolder = "./assets/classes/segment"
    # tamanho da janela
    windowSizes = [128, 64, 32, 16, 8, 4, 2]

    for windowSize in windowSizes:

        # inicialização do resultado
        resultArenito = []
        resultBasalto = []

        index = 0

        filters = [
            # # raw
            {"function": lambda x: np.mean(x[:, 0]), "label": "raw_mean"},
            {"function": lambda x: np.std(x[:, 0]), "label": "raw_std"},
            # # laplacianCv2
            # {"function": lambda x: np.mean(x[:, 1]), "label": "laplacianCv2_mean"},
            {"function": lambda x: np.std(x[:, 1]), "label": "laplacianCv2_std"},
            # # laplacian4
            # {"function": lambda x: np.mean(x[:, 2]), "label": "laplacian4_mean"},
            # {"function": lambda x: np.std(x[:, 2]), "label": "laplacian4_std"},
            # # laplacian8
            # {"function": lambda x: np.mean(x[:, 3]), "label": "laplacian8_mean"},
            # {"function": lambda x: np.std(x[:, 3]), "label": "laplacian8_std"},
            # # sobelMag
            # {"function": lambda x: np.mean(x[:, 4]), "label": "sobelMag_mean"},
            {"function": lambda x: np.std(x[:, 4]), "label": "sobelMag_std"},
            # # sobelAng
            # {"function": lambda x: np.mean(x[:, 5]), "label": "sobelAng_mean"},
            {"function": lambda x: np.std(x[:, 5]), "label": "sobelAng_std"},
            # # prewittMag
            # {"function": lambda x: np.mean(x[:, 6]), "label": "prewittMag_mean"},
            {"function": lambda x: np.std(x[:, 6]), "label": "prewittMag_std"},
            # # prewittAng
            # {"function": lambda x: np.mean(x[:, 7]), "label": "prewittAng_mean"},
            {"function": lambda x: np.std(x[:, 7]), "label": "prewittAng_std"},
            # # robertsMag
            # {"function": lambda x: np.mean(x[:, 8]), "label": "robertsMag_mean"},
            {"function": lambda x: np.std(x[:, 8]), "label": "robertsMag_std"},
            # # robertsAng
            # {"function": lambda x: np.mean(x[:, 9]), "label": "robertsAng_mean"},
            {"function": lambda x: np.std(x[:, 9]), "label": "robertsAng_std"},
            # # sobelAngToRoberts4
            {
                "function": lambda x: np.mean(x[:, 10]),
                "label": "sobelAngToRoberts4_mean",
            },
            {"function": lambda x: np.std(x[:, 10]), "label": "sobelAngToRoberts4_std"},
            # # sobelAngToRoberts8
            {
                "function": lambda x: np.mean(x[:, 11]),
                "label": "sobelAngToRoberts8_mean",
            },
            {"function": lambda x: np.std(x[:, 11]), "label": "sobelAngToRoberts8_std"},
            # # sobelAngToRoberts16
            {
                "function": lambda x: np.mean(x[:, 12]),
                "label": "sobelAngToRoberts16_mean",
            },
            {
                "function": lambda x: np.std(x[:, 12]),
                "label": "sobelAngToRoberts16_std",
            },
            # # sobelAngToRoberts32
            {
                "function": lambda x: np.mean(x[:, 13]),
                "label": "sobelAngToRoberts32_mean",
            },
            {
                "function": lambda x: np.std(x[:, 13]),
                "label": "sobelAngToRoberts32_std",
            },
            # # sobelAngToRoberts64
            {
                "function": lambda x: np.mean(x[:, 14]),
                "label": "sobelAngToRoberts64_mean",
            },
            {
                "function": lambda x: np.std(x[:, 14]),
                "label": "sobelAngToRoberts64_std",
            },
            # # sobelAngToRoberts128
            {
                "function": lambda x: np.mean(x[:, 15]),
                "label": "sobelAngToRoberts128_mean",
            },
            {
                "function": lambda x: np.std(x[:, 15]),
                "label": "sobelAngToRoberts128_std",
            },
        ]

        # itera pelos arquivos
        for folder, subfolders, files in os.walk(segmentedAreaFolder):
            for file in files:
                # print('reading segmentedArea file "{}"...'.format(file))
                segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
                classification = file[:7]

                # pegar tamanho pelo nome do arquivo
                diameter = file[10:-4]
                for i in range(len(diameter) - 1, -1, -1):
                    if diameter[i] == "-":
                        diameter = diameter[i + 1 :]
                diameter = int(diameter)

                if windowSize > diameter:
                    continue

                # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
                for windowVerticalIndex in range(int(np.floor(diameter / windowSize))):
                    temp = segmentedAreaDF.values[
                        diameter
                        * windowVerticalIndex
                        * windowSize : diameter
                        * windowVerticalIndex
                        * windowSize
                        + windowSize * diameter,
                        :,
                    ]
                    for windowHorizontalIndex in range(
                        int(np.floor(diameter / windowSize))
                    ):
                        
                        # permite apenas ímpares -> 1010
                        index += 1
                        if index % 2 == 0:
                            continue
                        
                        subSegmentedArea = []
                        for verticalIndex in range(windowSize):
                            subSegmentedArea.append(
                                temp[
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
                        data = [
                            filter["function"](subSegmentedArea) for filter in filters
                        ]
                        if classification == "arenito":
                            resultArenito.append(data)
                        else:
                            resultBasalto.append(data)

        # geração do dataframe das classes
        resultArenito = np.array(resultArenito)
        resultBasalto = np.array(resultBasalto)
        segmentedAreasResult = np.vstack((resultArenito, resultBasalto))
        mins = [np.min(segmentedAreasResult[:, i]) for i in range(segmentedAreasResult.shape[1])]
        maxs = [np.max(segmentedAreasResult[:, i]) for i in range(segmentedAreasResult.shape[1])]
        mins.append("min")
        maxs.append("max")

        arenito = [np.mean(resultArenito[:, i]) for i in range(resultArenito.shape[1])]
        basalto = [np.mean(resultBasalto[:, i]) for i in range(resultBasalto.shape[1])]
        arenito.append("arenito")
        basalto.append("basalto")

        header = [filter["label"] for filter in filters]
        header.append("class")
        header = np.array(header)

        resultDF = pd.DataFrame([arenito, basalto, mins, maxs], None, header)
        print(resultDF)

        # salvamento do arquivo
        filename = "classes_{}_odd_normalized.csv".format(windowSize)
        resultDF.to_csv(os.path.join(classesFolder, filename), index=False)


if __name__ == "__main__":
    main()
