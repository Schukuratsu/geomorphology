import numpy as np
import pandas as pd

filesList = [
    # "./assets/histograms/Arenito(raw).csv",
    # "./assets/histograms/Arenito(laplacian4).csv",
    "./assets/histograms/Arenito(laplacian8).csv",
    # "./assets/histograms/Basalto(raw).csv",
    # "./assets/histograms/Basalto(laplacian4).csv",
    "./assets/histograms/Basalto(laplacian8).csv",
]

meansFile = "./assets/means/histogram.csv"
resultsFile = "./assets/k-means/histogram.csv"


def getClosestClass(classItems, classLabels, entryData):
    results = np.zeros(classItems.shape[1])
    for classIndex in range(classItems.shape[1]):
        for attributeIndex in range(classItems.shape[0]):
            results[classIndex] = results[classIndex] + np.square(
                classItems[attributeIndex, classIndex] - entryData[attributeIndex]
            )
        results[classIndex] = np.sqrt(results[classIndex])
    resultingClass = np.argmin(results)
    return resultingClass, classLabels[resultingClass], results


def main():

    dataRange = [4,1004]

    meansDF = pd.read_csv(meansFile)
    meansValues = meansDF.values
    classItems = np.array([meansValues[dataRange[0]:dataRange[1], 3], meansValues[dataRange[0]:dataRange[1], 6]]).T
    classLabels = np.array(
        [meansDF.columns.values[1], meansDF.columns.values[4]]
    )

    index = []
    data = []

    for file in filesList:
        df = pd.read_csv(file)
        values = df.values
        for item in df.columns.values[1:]:
            index.append(item) 
        print(df.columns.values[1:])
        for i in range(1, values.shape[1]):
            classIdx, classLabel, teste = getClosestClass(classItems, classLabels, values[dataRange[0]:dataRange[1], i])
            print(teste, teste[0]/teste[1])
            data.append(classLabel)

    indexedData = np.array([index, data]).T
    print(indexedData)
    indexedDataDF = pd.DataFrame(indexedData)
    indexedDataDF.to_csv(resultsFile, header=False, index=False)


if __name__ == "__main__":
    main()
