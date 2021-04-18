import numpy as np
import pandas as pd


def main():

    filesList = np.array(
        [
            {"input": "./assets/histograms/Arenito(raw).csv", "label": "Arenito(raw)"},
            {
                "input": "./assets/histograms/Arenito(laplacian4).csv",
                "label": "Arenito(laplacian4)",
            },
            {
                "input": "./assets/histograms/Arenito(laplacian8).csv",
                "label": "Arenito(laplacian8)",
            },
            {"input": "./assets/histograms/Basalto(raw).csv", "label": "Basalto(raw)"},
            {
                "input": "./assets/histograms/Basalto(laplacian4).csv",
                "label": "Basalto(laplacian4)",
            },
            {
                "input": "./assets/histograms/Basalto(laplacian8).csv",
                "label": "Basalto(laplacian8)",
            },
        ]
    )

    means = np.empty((1004, 6))
    header = np.empty(filesList.shape[0] + 1, dtype=np.chararray)
    header[0] = "index"
    for filesIndex in range(filesList.shape[0]):
        header[filesIndex + 1] = filesList[filesIndex]["label"]
        df = pd.read_csv(filesList[filesIndex]["input"])
        values = df.values
        means[:, filesIndex] = np.mean(values[:, 1:], axis=1)

    index = np.reshape(values[:, 0],(1004,1))
    indexedMeans = np.hstack((index, means))
    indexedMeansDF = pd.DataFrame(indexedMeans)
    indexedMeansDF.to_csv("./assets/means/histogram.csv", header=header, index=False)


if __name__ == "__main__":
    main()
