from .DEM import DEM
import os
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def main():

    # configurações
    segmentedAreaFolder = "./assets/segmentedAreas/borderless"
    # tamanho da janela
    windowSize = 128

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
                        {
                            "class": classification,

                            ## MEANS ##
                            "raw_mean": np.mean(subSegmentedArea[:, 0]),

                            # "laplacianCv2_mean": np.mean(subSegmentedArea[:, 1]),
                            # "laplacian4_mean": np.mean(subSegmentedArea[:, 2]),
                            # "laplacian8_mean": np.mean(subSegmentedArea[:, 3]),

                            # "sobelMag_mean": np.mean(subSegmentedArea[:, 4]),
                            # "prewittMag_mean": np.mean(subSegmentedArea[:, 6]),
                            "robertsMag_mean": np.mean(subSegmentedArea[:, 8]),

                            # "sobelAng_mean": np.mean(subSegmentedArea[:, 5]),
                            # "prewittAng_mean": np.mean(subSegmentedArea[:, 7]),
                            # "robertsAng_mean": np.mean(subSegmentedArea[:, 9]),

                            # ## STDS ##
                            "raw_std": np.std(subSegmentedArea[:, 0]),

                            "laplacianCv2_std": np.std(subSegmentedArea[:, 1]),
                            # "laplacian4_std": np.std(subSegmentedArea[:, 2]),
                            # "laplacian8_std": np.std(subSegmentedArea[:, 3]),

                            # "sobelMag_std": np.std(subSegmentedArea[:, 4]),
                            # "prewittMag_std": np.std(subSegmentedArea[:, 6]),
                            # "robertsMag_std": np.std(subSegmentedArea[:, 8]),

                            # "sobelAng_std": np.std(subSegmentedArea[:, 5]),
                            # "prewittAng_std": np.std(subSegmentedArea[:, 7]),
                            # "robertsAng_std": np.std(subSegmentedArea[:, 9]),
                        }
                    )

    resultDF = pd.DataFrame(result)
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
