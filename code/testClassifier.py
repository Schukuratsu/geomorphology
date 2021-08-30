from .DEM import DEM
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def main():

    # configurações
    demFileName = "./assets/dems/S028-030_W053-055.tif"
    classesFolder = "./assets/classes/"
    # tamanho da janela

    # inicialização do resultado
    resultArenito = []
    resultBasalto = []

    index = 0

    filters = [
        {
            "label": "raw_std",
            "windowSize": 128,
            "file": "borderless/area/classes_128_odd_normalized.csv",
            "function": lambda x: np.std(x),
            "compare": lambda value, classes, classIndex: np.abs(
                (value - classes[classIndex, 1]) / classes[3, 1]
            ),
        },
    ]

    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()

    results = []
    for filter in filters:
        filteredHeightMap = filter["function"](demHeightMap)
        results.append(filteredHeightMap)

    smallestWindowSize = np.min([filter["windowSize"] for filter in filters])
    for i in range(0,demHeightMap.shape[0],smallestWindowSize):
        for j in range(0,demHeightMap.shape[1],smallestWindowSize):
            distances = []
            for index in range(len(results)):
                distances.append(filters[index]["compare"]())






    # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
    for windowVerticalIndex in range(int(np.floor(demHeightMap.shape[0] / windowSize))):
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
            data = [filter["function"](subSegmentedArea) for filter in filters]
            if classification == "arenito":
                resultArenito.append(data)
            else:
                resultBasalto.append(data)

    # classificação
    inputFile = os.path.join(classesFolder, "classes_{}_odd.csv".format(windowSize))
    classesDF = pd.read_csv(inputFile)

    classificationHeader = [
        "original_class",
        "classified_as",
        "%_of_certainty",
        "dist_arenito",
        "dist_basalto",
    ]

    classificationResults = []

    for area in resultArenito:
        distancesArenito = [
            filters[i]["compare"](
                area[i], classesDF.values[0, :], filters[i]["multiplier"]
            )
            for i in range(len(filters))
        ]
        distancesBasalto = [
            filters[i]["compare"](
                area[i], classesDF.values[1, :], filters[i]["multiplier"]
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
        pOfCertainty = (dBasalto if classified_as == "arenito" else dArenito) / (
            dArenito + dBasalto
        )
        classificationResults.append(
            [
                "arenito",
                classified_as,
                pOfCertainty,
                dArenito,
                dBasalto,
            ]
        )

    for area in resultBasalto:
        distancesArenito = [
            filters[i]["compare"](
                area[i], classesDF.values[0, :], filters[i]["multiplier"]
            )
            for i in range(len(filters))
        ]
        distancesBasalto = [
            filters[i]["compare"](
                area[i], classesDF.values[1, :], filters[i]["multiplier"]
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
        pOfCertainty = (dBasalto if classified_as == "arenito" else dArenito) / (
            dArenito + dBasalto
        )
        classificationResults.append(
            [
                "basalto",
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
