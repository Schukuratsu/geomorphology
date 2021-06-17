from .DEM import DEM
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def normalize(line, mins, maxs):
    return np.divide(np.subtract(line, mins), maxs)


def main():

    # configurações
    segmentedAreaFolder = "./assets/segmentedAreas"
    classesFolder = "./assets/classes/segment"
    classificationFolder = "./assets/classifications"
    # tamanho da janela
    windowSizes = [128, 64, 32, 16, 8, 4, 2]

    for windowSize in windowSizes:

        # inicialização do resultado
        resultArenito = []
        resultBasalto = []

        index = 0

        filters = [
            # # raw
            # {
            #     "label": "raw_mean",
            #     "function": lambda x: np.mean(x[:, 0]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 0])/classes[3, 0] * multiplier
            #     ),
            # },
            {
                "label": "raw_std",
                "function": lambda x: np.std(x[:, 0]),
                "multiplier": 1,
                "compare": lambda value, classes, classIndex, multiplier: np.abs(
                    (value - classes[classIndex, 1]) / classes[3, 1] * multiplier
                ),
            },
            # # # laplacianCv2
            # {
            #     "label": "laplacianCv2_std",
            #     "function": lambda x: np.std(x[:, 1]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 2])/classes[3, 2] * multiplier
            #     ),
            # },
            # # sobelMag
            {
                "label": "sobelMag_std",
                "function": lambda x: np.std(x[:, 4]),
                "multiplier": 1,
                "compare": lambda value, classes, classIndex, multiplier: np.abs(
                    (value - classes[classIndex, 3]) / classes[3, 3] * multiplier
                ),
            },
            # # # sobelAng
            # {
            #     "label": "sobelAng_std",
            #     "function": lambda x: np.std(x[:, 5]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 4])/classes[3, 4] * multiplier
            #     ),
            # },
            # # # prewittMag
            # {
            #     "label": "prewittMag_std",
            #     "function": lambda x: np.std(x[:, 6]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 5])/classes[3, 5] * multiplier
            #     ),
            # },
            # # # prewittAng
            # {
            #     "label": "prewittAng_std",
            #     "function": lambda x: np.std(x[:, 7]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 6])/classes[3, 6] * multiplier
            #     ),
            # },
            # # # robertsMag
            # {
            #     "label": "robertsMag_std",
            #     "function": lambda x: np.std(x[:, 8]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 7])/classes[3, 7] * multiplier
            #     ),
            # },
            # # # robertsAng
            # {
            #     "label": "robertsAng_std",
            #     "function": lambda x: np.std(x[:, 9]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 8])/classes[3, 8] * multiplier
            #     ),
            # },
            # # # sobelAngToRoberts4
            # {
            #     "label": "sobelAngToRoberts4_mean",
            #     "function": lambda x: np.mean(x[:, 10]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 9])/classes[3, 9] * multiplier
            #     ),
            # },
            # {
            #     "label": "sobelAngToRoberts4_std",
            #     "function": lambda x: np.std(x[:, 10]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 10])/classes[3, 10] * multiplier
            #     ),
            # },
            # # # sobelAngToRoberts8
            # {
            #     "label": "sobelAngToRoberts8_mean",
            #     "function": lambda x: np.mean(x[:, 11]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 11])/classes[3, 11] * multiplier
            #     ),
            # },
            # {
            #     "label": "sobelAngToRoberts8_std",
            #     "function": lambda x: np.std(x[:, 11]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 12])/classes[3, 12] * multiplier
            #     ),
            # },
            # # # sobelAngToRoberts16
            # {
            #     "label": "sobelAngToRoberts16_mean",
            #     "function": lambda x: np.mean(x[:, 12]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 13])/classes[3, 13] * multiplier
            #     ),
            # },
            # {
            #     "label": "sobelAngToRoberts16_std",
            #     "function": lambda x: np.std(x[:, 12]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 14])/classes[3, 14] * multiplier
            #     ),
            # },
            # # # sobelAngToRoberts32
            # {
            #     "label": "sobelAngToRoberts32_mean",
            #     "function": lambda x: np.mean(x[:, 13]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 15])/classes[3, 15] * multiplier
            #     ),
            # },
            # {
            #     "label": "sobelAngToRoberts32_std",
            #     "function": lambda x: np.std(x[:, 13]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 16])/classes[3, 16] * multiplier
            #     ),
            # },
            # # # sobelAngToRoberts64
            # {
            #     "label": "sobelAngToRoberts64_mean",
            #     "function": lambda x: np.mean(x[:, 14]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 17])/classes[3, 17] * multiplier
            #     ),
            # },
            # {
            #     "label": "sobelAngToRoberts64_std",
            #     "function": lambda x: np.std(x[:, 14]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 18])/classes[3, 18] * multiplier
            #     ),
            # },
            # # # sobelAngToRoberts128
            # {
            #     "label": "sobelAngToRoberts128_mean",
            #     "function": lambda x: np.mean(x[:, 15]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 19])/classes[3, 19] * multiplier
            #     ),
            # },
            # {
            #     "label": "sobelAngToRoberts128_std",
            #     "function": lambda x: np.std(x[:, 15]),
            #     "multiplier": 1,
            #     "compare": lambda value, classes, classIndex, multiplier: np.abs(
            #         (value - classes[classIndex, 20])/classes[3, 20] * multiplier
            #     ),
            # },
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

                        # permite apenas pares -> 0101
                        index += 1
                        if index % 2 == 1:
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

        # classificação
        inputFile = os.path.join(
            classesFolder, "classes_{}_odd_normalized.csv".format(windowSize)
        )
        classesDF = pd.read_csv(inputFile)

        classificationHeader = [
            "original_class",
            "classified_as",
            "%_of_certainty",
            "dist_arenito",
            "dist_basalto",
        ]

        classificationResults = []

        step = [
            {"data": resultArenito, "label": "arenito"},
            {"data": resultBasalto, "label": "basalto"},
        ]

        for part in step:
            for area in part["data"]:
                distancesArenito = [
                    filters[i]["compare"](
                        area[i],
                        classesDF.values,
                        0,
                        filters[i]["multiplier"],
                    )
                    for i in range(len(filters))
                ]
                distancesBasalto = [
                    filters[i]["compare"](
                        area[i],
                        classesDF.values,
                        1,
                        filters[i]["multiplier"],
                    )
                    for i in range(len(filters))
                ]
                # aqui não está implementando distância espacial, testar
                # dArenito = np.sum(np.abs(distancesArenito))
                # dBasalto = np.sum(np.abs(distancesBasalto))
                dArenito = np.sqrt(np.sum(np.square(distancesArenito)))
                dBasalto = np.sqrt(np.sum(np.square(distancesBasalto)))
                classified_as = (
                    "arenito"
                    if dArenito < dBasalto
                    else ("basalto" if dArenito > dBasalto else "nenhum")
                )
                pOfCertainty = (
                    dBasalto if classified_as == "arenito" else dArenito
                ) / (dArenito + dBasalto)
                classificationResults.append(
                    [
                        part["label"],
                        classified_as,
                        pOfCertainty,
                        dArenito,
                        dBasalto,
                    ]
                )

        arenitoMiss = 0
        basaltoMiss = 0
        for area in classificationResults:
            if area[0] != area[1]:
                if area[0] == "arenito":
                    arenitoMiss += 1
                else:
                    basaltoMiss += 1

        qtdArenito = len(resultArenito)
        qtdBasalto = len(resultBasalto)

        pAcertoArenito = 1 - arenitoMiss / qtdArenito
        pAcertoBasalto = 1 - basaltoMiss / qtdBasalto
        pAcertoGeral = 1 - (arenitoMiss + basaltoMiss) / (qtdArenito + qtdBasalto)
        print(windowSize)
        print("% acerto arenito: {:.2f}".format(pAcertoArenito))
        print("% acerto basalto: {:.2f}".format(pAcertoBasalto))
        print("% acerto geral: {:.2f}\n".format(pAcertoGeral))

        # geração do dataframe das classes
        classificationDF = pd.DataFrame(
            np.array(classificationResults), None, classificationHeader
        )
        # print(classificationDF)

        # salvamento do arquivo
        outputFile = os.path.join(
            classificationFolder, "classification_{}_odd.csv".format(windowSize)
        )
        classificationDF.to_csv(outputFile, index=False)


if __name__ == "__main__":
    main()
