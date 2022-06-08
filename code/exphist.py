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
    # for windowSize in [8, 16, 32, 64, 128]:
    for windowSize in [2, 4]:
        models = [[], [], []]
        classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)

        fileCount = -1
        for folder, subfolders, files in os.walk(segmentedAreaFolder):
            for file in files:
                print(file)
                fileCount += 1
                # if fileCount > 1:
                #     break

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
                    continue

                # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
                for windowVerticalIndex in range(int(np.floor(diameter - windowSize))):
                    # recorta linhas a serem utilizadas
                    temp = segmentedAreaDF.values[
                        diameter * windowVerticalIndex : diameter * windowVerticalIndex
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
                                    + windowHorizontalIndex : diameter * verticalIndex
                                    + windowHorizontalIndex
                                    + windowSize,
                                    :,
                                ]
                            )

                        step = subSegmentedArea[0]
                        for sub in subSegmentedArea[1:]:
                            step = np.vstack((step, sub))
                        subSegmentedArea = step

                        for index in range(length):
                            if modelMasks[index][fileCount % length] == True:

                                # # gera um valor para cada banda final
                                models[index].append(
                                    {
                                        "file": file,
                                        "mean": np.mean(subSegmentedArea[:, 0]),
                                        "std": np.std(subSegmentedArea[:, 0]),
                                        "roberts_mag_mean": np.mean(
                                            subSegmentedArea[:, 8]
                                        ),
                                        "laplacian_cv2_std": np.std(
                                            subSegmentedArea[:, 1]
                                        ),
                                        "class": classification,
                                    }
                                )

        # header = ["file", "mean", "std", "class"]
        for cn in ["arenito", "basalto"]:
            for model in range(length):
                resultDF = pd.DataFrame(models[model])
                array = resultDF.values[resultDF["class"].values == cn]
                arrayDF = pd.DataFrame(array)
                arrayDF.to_csv(
                    os.path.join(classesFolder, "{}_{}.csv".format(cn, model)),
                    index=None,
                )


def knn1geraTudo():
    """gera 3 modelos baseados em elementos diferentes para cada classe"""
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    for windowSize in [32]:
        model = []
        classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)

        for folder, subfolders, files in os.walk(segmentedAreaFolder):
            for file in files:
                print(file)

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
                    continue

                # iterar pelo dataset varias vezes a fim de pegar os dados de cada janela
                for windowVerticalIndex in range(int(np.floor(diameter - windowSize))):
                    # recorta linhas a serem utilizadas
                    temp = segmentedAreaDF.values[
                        diameter * windowVerticalIndex : diameter * windowVerticalIndex
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
                                    + windowHorizontalIndex : diameter * verticalIndex
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
                        model.append(
                            {
                                "file": file,
                                "mean": np.mean(subSegmentedArea[:, 0]),
                                "std": np.std(subSegmentedArea[:, 0]),
                                "roberts_mag_mean": np.mean(subSegmentedArea[:, 8]),
                                "laplacian_cv2_std": np.std(subSegmentedArea[:, 1]),
                                "class": classification,
                            }
                        )

        resultDF = pd.DataFrame(model)
        resultDF.to_csv(
            os.path.join(classesFolder, "geral.csv"),
            index=None,
        )


def knn2():
    """gera 3 resultados baseados nos 3 modelos gerados pela função anterior (variáveis são windowSize, modelNumber e className)"""
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
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
    # for windowSize in [8, 16, 32, 64, 128]:
    for windowSize in [2, 4]:
        models = [[], [], []]
        classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)

        fileCount = -1
        for folder, subfolders, files in os.walk(segmentedAreaFolder):
            for file in files:
                print(file)
                fileCount += 1
                # if fileCount > 1:
                #     break

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

                        for index in range(length):
                            if modelMasks[index][fileCount % length] == False:

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
            resultDF = pd.DataFrame(models[model])
            arenitoDF = pd.read_csv(
                os.path.join(classesFolder, "{}_{}.csv".format("arenito", model))
            )
            basaltoDF = pd.read_csv(
                os.path.join(classesFolder, "{}_{}.csv".format("basalto", model))
            )

            basalto_std = basaltoDF["std"].mean()
            arenito_std = arenitoDF["std"].mean()

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
                    certainty = 1 - np.abs(diffBasalto / (diffArenito + diffBasalto))
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
            print(classificationDF)
            classificationDF.to_csv(
                os.path.join(classesFolder, "result_{}.csv".format(model)), index=None
            )


def knn2ponto5():
    """classificador com upsamplig simples"""
    secret = 2
    for windowSize in [2]:
    # for windowSize in [2, 4, 8, 16, 32, 64, 128]:
        print("\n\njanela: {}".format(windowSize))
        classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)

        # dataset com todos os dados
        dataset = pd.read_csv(os.path.join(classesFolder, "{}.csv".format("geral")))

        # train = pra gerar o modelo; test = pra avaliar o modelo
        from sklearn.model_selection import train_test_split

        dataset_train, dataset_test = train_test_split(dataset, test_size=0.3)

        # upsampling do arenito
        from sklearn.utils import resample

        arenito_train = dataset_train[dataset_train["class"] == "arenito"]
        basalto_train = dataset_train[dataset_train["class"] != "arenito"]
        arenito_test = dataset_test[dataset_test["class"] == "arenito"]
        basalto_test = dataset_test[dataset_test["class"] != "arenito"]
        arenito_train_up = resample(
            arenito_train,
            random_state=36,
            n_samples=basalto_train.values.shape[0],
            replace=True,
        )
        arenito_test_up = resample(
            arenito_test,
            random_state=36,
            n_samples=basalto_test.values.shape[0],
            replace=True,
        )
        dataset_train_up = pd.concat([arenito_train_up, basalto_train])
        dataset_test_up = pd.concat([arenito_test_up, basalto_test])

        # x = entrada; y = saída
        x_train = dataset_train_up.iloc[:, 1:-1].values
        y_train = dataset_train_up.iloc[:, 5].values
        x_test = dataset_test_up.iloc[:, 1:-1].values
        y_test = dataset_test_up.iloc[:, 5].values

        print('arenito_train:\n',arenito_train)
        print('basalto_train:\n',basalto_train)
        print('arenito_test:\n',arenito_test)
        print('basalto_test:\n',basalto_test)

        print('x_train:\n',x_train)
        print('y_train:\n',y_train)
        print('x_test:\n',x_test)
        print('y_test:\n',y_test)

        # ajustando o scaler (?)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # gera modelo de classificador KNN
        from sklearn.neighbors import KNeighborsClassifier

        classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        classifier.fit(x_train, y_train)

        # gera resultado
        y_pred = classifier.predict(x_test)

        # mostra resultado
        from sklearn.metrics import classification_report, confusion_matrix

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        # salva o modelo
        from joblib import dump

        dump(classifier, os.path.join(classesFolder, "classifier_{}.joblib".format(secret)))


def knn2ponto6():
    """classificador com upsampling separando as áreas seguimentadas"""
    for windowSize in [2, 4, 8, 16, 32, 64, 128]:
        print("\n\njanela: {}".format(windowSize))
        classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)

        # dataset com todos os dados
        dataset = pd.read_csv(os.path.join(classesFolder, "{}.csv".format("geral")))

        # train = pra gerar o modelo; test = pra avaliar o modelo
        from sklearn.model_selection import train_test_split

        dataset_arenito = dataset[dataset["class"] == "arenito"]
        dataset_basalto = dataset[dataset["class"] == "basalto"]

        files_arenito = list(set(dataset_arenito['file'].values))
        files_basalto = list(set(dataset_basalto['file'].values))

        files_arenito_train, files_arenito_test = train_test_split(files_arenito, test_size=0.3)
        files_basalto_train, files_basalto_test = train_test_split(files_basalto, test_size=0.3)

        # # upsampling do arenito
        from sklearn.utils import resample

        get_files_lines = lambda ds, files: [a in files for a in ds['file'].values]

        arenito_train = dataset[get_files_lines(dataset, files_arenito_train)]
        basalto_train = dataset[get_files_lines(dataset,files_basalto_train)]
        arenito_test = dataset[get_files_lines(dataset,files_arenito_test)]
        basalto_test = dataset[get_files_lines(dataset,files_basalto_test)]
        arenito_train_up = resample(
            arenito_train,
            random_state=36,
            n_samples=basalto_train.values.shape[0],
            replace=True,
        )
        arenito_test_up = resample(
            arenito_test,
            random_state=36,
            n_samples=basalto_test.values.shape[0],
            replace=True,
        )
        dataset_train_up = pd.concat([arenito_train_up, basalto_train])
        dataset_test_up = pd.concat([arenito_test_up, basalto_test])

        # x = entrada; y = saída
        x_train = dataset_train_up.iloc[:, 1:-1].values
        y_train = dataset_train_up.iloc[:, 5].values
        x_test = dataset_test_up.iloc[:, 1:-1].values
        y_test = dataset_test_up.iloc[:, 5].values

        # print('arenito_train:\n',arenito_train)
        # print('basalto_train:\n',basalto_train)
        # print('arenito_test:\n',arenito_test)
        # print('basalto_test:\n',basalto_test)

        # print('x_train:\n',x_train)
        # print('y_train:\n',y_train)
        # print('x_test:\n',x_test)
        # print('y_test:\n',y_test)

        # ajustando o scaler (?)
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # gera modelo de classificador KNN
        from sklearn.neighbors import KNeighborsClassifier

        classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        classifier.fit(x_train, y_train)

        # gera resultado
        y_pred = classifier.predict(x_test)

        # mostra resultado
        from sklearn.metrics import classification_report, confusion_matrix

        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        # salva o modelo
        from joblib import dump

        dump(classifier, os.path.join(classesFolder, "classifier{}.joblib".format(windowSize)))

# use this one
def knn2ponto7():
    """classificador com downsamplig simples"""
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from joblib import dump
    secret = 2
    # for windowSize in [2, 4, 8, 16, 32, 64, 128]:
    for windowSize in [16]:
        print("\n\njanela: {}".format(windowSize))
        classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)

        # dataset com todos os dados
        dataset = pd.read_csv(os.path.join(classesFolder, "{}.csv".format("geral")))

        # train = pra gerar o modelo; test = pra avaliar o modelo
        dataset_train, dataset_test = train_test_split(dataset, test_size=0.3)

        # upsampling do arenito
        arenito_train = dataset_train[dataset_train["class"] == "arenito"]
        basalto_train = dataset_train[dataset_train["class"] != "arenito"]
        arenito_test = dataset_test[dataset_test["class"] == "arenito"]
        basalto_test = dataset_test[dataset_test["class"] != "arenito"]
        basalto_train_down = resample(
            basalto_train,
            random_state=36,
            n_samples=arenito_train.values.shape[0],
            replace=True,
        )
        basalto_test_down = resample(
            basalto_test,
            random_state=36,
            n_samples=arenito_test.values.shape[0],
            replace=True,
        )
        dataset_train_down = pd.concat([arenito_train, basalto_train_down])
        dataset_test_down = pd.concat([arenito_test, basalto_test_down])

        # x = entrada; y = saída
        x_train = dataset_train_down.iloc[:, 1:-1].values
        y_train = dataset_train_down.iloc[:, 5].values
        x_test = dataset_test_down.iloc[:, 1:-1].values
        y_test = dataset_test_down.iloc[:, 5].values

        # print('arenito_train:\n',arenito_train)
        # print('basalto_train:\n',basalto_train)
        # print('arenito_test:\n',arenito_test)
        # print('basalto_test:\n',basalto_test)

        # print('x_train:\n',x_train)
        # print('y_train:\n',y_train)
        # print('x_test:\n',x_test)
        # print('y_test:\n',y_test)

        # ajustando o scaler (?)
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # gera modelo de classificador KNN
        classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        classifier.fit(x_train, y_train)

        # gera resultado
        y_pred = classifier.predict(x_test)

        # mostra resultado
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        # salva o modelo
        dump(classifier, os.path.join(classesFolder, "classifier{}_{}.joblib".format(windowSize, secret)))


def knn3():
    """gera resultados finais dos 3 modelos gerados pela função anterior (variáveis são windowSize e modelNumber)"""
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    result = []
    resultFolder = "./assets/classes/new/exphist"
    for windowSize in [2, 4]:
        # for windowSize in [2, 4, 8, 16, 32, 64, 128]:
        for model in [0, 1, 2]:
            classesFolder = "./assets/classes/new/exphist/{}".format(windowSize)
            classificationDF = pd.read_csv(
                os.path.join(classesFolder, "result_{}.csv".format(model))
            )
            classesValues = []
            for classes in [["arenito"], ["basalto"], ["arenito", "basalto"]]:
                # filtrar por classes
                classDF = classificationDF.loc[
                    classificationDF["trueClass"].isin(classes)
                ]
                # calcular taxa de acerto
                numAcertos = 0
                for linha in range(classDF.values.shape[0]):
                    if classDF.values[linha, 0] == classDF.values[linha, 1]:
                        numAcertos += 1
                numTotal = classDF.values.shape[0]
                taxaDeAcerto = 100 * numAcertos / numTotal
                # calcular taxa de certeza média
                taxaDeCertezaMedia = classDF["certainty"].mean() * 100
                print(classDF)
                print(numAcertos, numTotal, taxaDeAcerto, taxaDeCertezaMedia)
                classesValues.append(
                    {
                        "taxaDeAcerto": taxaDeAcerto,
                        "taxaDeCertezaMedia": taxaDeCertezaMedia,
                    }
                )
            result.append(
                {
                    "windowSize": windowSize,
                    "model": model,
                    "%acertoArenito": classesValues[0]["taxaDeAcerto"],
                    "%certezaArenito": classesValues[0]["taxaDeCertezaMedia"],
                    "%acertoBasalto": classesValues[1]["taxaDeAcerto"],
                    "%certezaBasalto": classesValues[1]["taxaDeCertezaMedia"],
                    "%acertoGeral": classesValues[2]["taxaDeAcerto"],
                    "%certezaGeral": classesValues[2]["taxaDeCertezaMedia"],
                }
            )
    resultDF = pd.DataFrame(result)
    resultDF.to_csv(os.path.join(resultFolder, "result.csv"), index=None)


def testeSciKit():

    # dataset com todos os dados
    dataset = pd.DataFrame(
        [[1, "impar"], [3, "impar"], [5, "impar"], [6, "par"], [4, "par"], [2, "par"]],
        None,
        ["val", "res"],
    )

    # x = entrada; y = saída
    x = dataset.iloc[:, 0].values.reshape(-1, 1)
    y = dataset.iloc[:, 1].values

    # train = pra gerar o modelo; test = pra avaliar o modelo
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.66)

    # ajustando o scaler (?)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # gera modelo de classificador KNN
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(x_train, y_train)

    # gera resultado
    y_pred = classifier.predict(x_test)

    # mostra resultado
    from sklearn.metrics import classification_report, confusion_matrix

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))