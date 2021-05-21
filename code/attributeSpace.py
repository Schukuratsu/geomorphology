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

    # inicialização do resultado
    result = []

    # itera pelos arquivos
    for folder, subfolders, files in os.walk(segmentedAreaFolder):
        for file in files:
            print('reading segmentedArea file "{}"...'.format(file))
            segmentedAreaDF = pd.read_csv(os.path.join(folder, file))
            classification = file[:7]

            # # gera um valor para cada banda final
            result.append(
                [
                    # "raw_std",
                    np.std(segmentedAreaDF.values[:, 0]),
                    # "sobelMag",
                    np.mean(segmentedAreaDF.values[:, 4]),
                    # # "sobelMag_std",
                    # np.std(segmentedAreaDF.values[:, 4]),
                    # # "sobelAng",
                    # np.mean(segmentedAreaDF.values[:, 5]),
                    # # "sobelAng_std",
                    # np.std(segmentedAreaDF.values[:, 5]),
                    # # "robertsAng",
                    # np.mean(segmentedAreaDF.values[:, 9]),
                    # # "robertsAng_std",
                    # np.std(segmentedAreaDF.values[:, 9]),
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
            # "sobelAng",
            # "sobelAng_std",
            # "robertsAng",
            # "robertsAng_std",
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
