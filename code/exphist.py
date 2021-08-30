from .DEM import DEM
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd


def printDetails(arr):
    aaa = [
        np.mean(arr),
        np.min(arr),
        np.max(arr),
        np.std(arr),
        np.percentile(arr, 25),
        np.percentile(arr, 50),
        np.percentile(arr, 75),
    ]
    print(
        "média: {}\nmin: {}\nmax: {}\ndesvio: {}\n1ºquartil: {}\nmediana: {}\n3ºquartil: {}\n".format(
            *aaa
        )
    )
    return aaa


def regiao1():
    dem = DEM("./assets/dems/S028-030_W053-055.tif")
    hm = dem.getNumpyHeightMap()

    sns.histplot(hm.ravel(), binwidth=1)
    plt.show()


def regiao3():
    dem = DEM("./assets/dems/areadeestudo.tif")
    hm = dem.getNumpyHeightMap()

    validValues = []
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            if hm[i, j] != -1:
                validValues.append(hm[i, j])
    printDetails(validValues)

    sns.histplot(hm.ravel(), binrange=(0, 1000), binwidth=1)
    plt.show()


def classes1():
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    arenito = []
    basalto = []

    for folder, subfolders, files in os.walk(segmentedAreaFolder):
        for file in files:
            segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
            classification = file[:7]
            if classification == "arenito":
                arenito.append(segmentedAreaDF.values[:, 0])
            else:
                basalto.append(segmentedAreaDF.values[:, 0])
    steparenito = [x for array in arenito for x in array]
    stepbasalto = [x for array in basalto for x in array]
    printDetails("arenito")
    printDetails(steparenito)
    printDetails("basalto")
    printDetails(stepbasalto)
    step = [
        {"value": value, "class": classLabel}
        for classArray, classLabel in [(arenito, "arenito"), (basalto, "basalto")]
        for array in classArray
        for value in array
    ]
    df = pd.DataFrame(step)
    print(df)
    sns.histplot(df, x="value", hue="class", binwidth=1)
    plt.show()


def classes2():
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    arenito = []
    basalto = []

    for folder, subfolders, files in os.walk(segmentedAreaFolder):
        for file in files:
            segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
            classification = file[:7]
            if classification == "arenito":
                arenito.append(segmentedAreaDF.values[:, 0])
            else:
                basalto.append(segmentedAreaDF.values[:, 0])
    step = [
        {"value": value, "class": classLabel}
        for classArray, classLabel in [
            (arenito, "arenito"),
            (basalto, "basalto"),
        ]
        for array in classArray
        for value in array
    ]
    df = pd.DataFrame(step)
    print(df)
    sns.displot(
        df, x="value", hue="class", binwidth=1, stat="probability", common_norm=False
    )
    plt.show()


def class1():
    windowSize = 32
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    result = []

    for folder, subfolders, files in os.walk(segmentedAreaFolder):
        for file in files:
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
                    result.append(
                        [
                            file,
                            np.mean(subSegmentedArea[:, 0]),
                            np.min(subSegmentedArea[:, 0]),
                            np.max(subSegmentedArea[:, 0]),
                            np.std(subSegmentedArea[:, 0]),
                            np.percentile(subSegmentedArea[:, 0], 25),
                            np.percentile(subSegmentedArea[:, 0], 50),
                            np.percentile(subSegmentedArea[:, 0], 75),
                            classification,
                        ]
                    )

    header = [
        "file",
        "mean",
        "min",
        "max",
        "std",
        "percentile 25",
        "percentile 50",
        "percentile 75",
        "class",
    ]
    resultDF = pd.DataFrame(result, None, header)
    print(resultDF)

    arenito = []
    basalto = []
    for i in range(resultDF.values.shape[0]):
        if resultDF.values[i, 8] == "arenito":
            arenito.append(resultDF.values[i, 4])
        else:
            basalto.append(resultDF.values[i, 4])

    print("std " + str(windowSize) + "arenito/basalto")
    printDetails(arenito)
    printDetails(basalto)
    sns.displot(
        resultDF,
        x="std",
        hue="class",
        binwidth=1,
        stat="probability",
        common_norm=False,
    )
    plt.show()


def knn1():
    """gera 3 modelos baseados em elementos diferentes para cada classe"""
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    for windowSize in [8,16,32,64,128]:
        classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)
        models = [[], [], []]
        modelMasks = [
            [
                True,
                False,
                False,
            ],
            [
                False,
                True,
                False,
            ],
            [
                False,
                False,
                True,
            ],
        ]
        length = len(modelMasks)

        fileCount = -1
        for folder, subfolders, files in os.walk(segmentedAreaFolder):
            for file in files:
                print(file)
                fileCount += 1
                # if fileCount > 1:
                #     break
                for index in range(length):
                    if modelMasks[index][fileCount % length] == True:

                        segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
                        classification = file[:7]

                        # pegar tamanho pelo nome do arquivo
                        diameter = file[10:-4]
                        for i in range(len(diameter) - 1, -1, -1):
                            if diameter[i] == "-":
                                diameter = diameter[i + 1 :]
                        diameter = int(diameter)

                        if windowSize > diameter:
                            print("muy peqeno")
                            break

                        # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
                        for windowVerticalIndex in range(
                            int(np.floor(diameter - windowSize))
                        ):
                            # recorta linhas a serem utilizadas
                            temp = segmentedAreaDF.values[
                                diameter
                                * windowVerticalIndex : diameter
                                * windowVerticalIndex
                                + windowSize * diameter,
                                :,
                            ]
                            for windowHorizontalIndex in range(
                                int(np.floor(diameter - windowSize))
                            ):
                                subSegmentedArea = []
                                # recorta colunas a serem utilizadas
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
                                models[index].append(
                                    {
                                        "file": file,
                                        "mean": np.mean(subSegmentedArea[:, 0]),
                                        "std": np.std(subSegmentedArea[:, 0]),
                                        "roberts_mag_mean": np.mean(subSegmentedArea[:, 8]),
                                        "laplacian_cv2_std": np.std(subSegmentedArea[:, 1]),
                                        "class": classification,
                                    }
                                )

        header = ["file", "mean", "std", "class"]
        for cn in ["arenito","basalto"]:
            for model in range(length):
                resultDF = pd.DataFrame(models[model], None, header)
                array = resultDF.values[resultDF["class"].values == cn]
                arrayDF = pd.DataFrame(array, None, header)
                arrayDF.to_csv(
                    os.path.join(classesFolder, "{}_{}.csv".format(cn, model)), index=None
                )


def knn2():
    """gera 3 modelos baseados em elementos diferentes para cada classe"""
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    windowSize = 8
    classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)
    models = [[], [], []]
    modelMasks = [
        [
            True,
            False,
            False,
        ],
        [
            False,
            True,
            False,
        ],
        [
            False,
            False,
            True,
        ],
    ]
    length = len(modelMasks)

    fileCount = -1
    for folder, subfolders, files in os.walk(segmentedAreaFolder):
        for file in files:
            print(file)
            fileCount += 1
            # if fileCount > 1:
            #     break
            for index in range(length):
                if modelMasks[index][fileCount % length] == False:

                    segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
                    classification = file[:7]

                    # pegar tamanho pelo nome do arquivo
                    diameter = file[10:-4]
                    for i in range(len(diameter) - 1, -1, -1):
                        if diameter[i] == "-":
                            diameter = diameter[i + 1 :]
                    diameter = int(diameter)

                    if windowSize > diameter:
                        print("muy peqeno")
                        break

                    # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
                    for windowVerticalIndex in range(
                        int(np.floor(diameter / windowSize))
                    ):
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
                            models[index].append(
                                {
                                    "file": file,
                                    "mean": np.mean(subSegmentedArea[:, 0]),
                                    "std": np.std(subSegmentedArea[:, 0]),
                                    "class": classification,
                                }
                            )

    classificationResults = [[], [], []]
    for model in range(length):
        resultDF = pd.DataFrame(models[model], None, header)
        arenitoDF = pd.read_csv(
            os.path.join(classesFolder, "{}_{}.csv".format("arenito", model))
        )
        basaltoDF = pd.read_csv(
            os.path.join(classesFolder, "{}_{}.csv".format("basalto", model))
        )

        basalto_std = basaltoDF.mean("std")
        arenito_std = arenitoDF.mean("std")

        for i in range(resultDF.values.shape[0]):
            line = resultDF.values[i, :]

            diffArenito = np.abs(line[2] - arenito_std)
            diffBasalto = np.abs(line[2] - basalto_std)
            trueClass = line[3]
            if diffArenito < diffBasalto:
                givenClass = "arenito"
                certainty = 1 - np.abs(diffArenito / (diffArenito + diffBasalto))
            else:
                givenClass = "basalto"
                certainty = 1 - np.abs(diffArenito / (diffArenito + diffBasalto))
            classificationResults[model].append(
                {
                    "trueClass": trueClass,
                    "givenClass": givenClass,
                    "certainty": certainty,
                    "diffArenito": diffArenito,
                    "diffBasalto": diffBasalto,
                }
            )

        classificationDF = pd.DataFrame(classificationResults[model])
