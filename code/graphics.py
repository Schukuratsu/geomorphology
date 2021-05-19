import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from .DEM import DEM
import numpy as np
import pandas as pd


def plotHist(arr):
    # if not limits:
    #     # avoid the -1 on finding the min value
    #     maxValue = np.max(arr)
    #     tempArr = np.copy(arr)
    #     tempArr[tempArr == -1] = maxValue
    #     minValue = np.min(tempArr)
    #     limits = [minValue, maxValue]
    # if not bins:
    #     bins = limits[1] - limits[0]
    sns.histplot(data=arr.ravel(), binwidth=1, kde=True)
    plt.show()


def main():

    print("initiating histogram extractor.")

    demFileName = "./assets/dems/S028-030_W053-055.tif"

    print('reading main DEM file "{}"...'.format(demFileName))
    dem = DEM(demFileName)
    demHeightMap = dem.getNumpyHeightMap()

    print('generating histogram...')
    plotHist(demHeightMap[4000:-4000,4000:-4000])
    # plotHist(demHeightMap, limits=options["limits"], show=True)

    if dem:
        dem.clear()


if __name__ == "__main__":
    main()
