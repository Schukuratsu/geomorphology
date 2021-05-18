from .DEM import DEM
import numpy as np
import pandas as pd


def getSimpleRadius(mat, center, nullValue=-1):
    yi, xi = center
    for x in range(xi, -1, -1):
        if mat[yi, x] == nullValue:
            return xi - x


def main():

    print("initiating radius extractor.")

    outputSeedsFileName = "./assets/seeds/centroides.csv"

    optionsArray = [
        {
            "segmentedFileName": "./assets/maps/Arenito recortado (completo).tif",
            "seedsFileName": "./assets/seeds/Centroides Arenito (completo).csv",
            "classification": "arenito",
        },
        {
            "segmentedFileName": "./assets/maps/Basalto recortado.tif",
            "seedsFileName": "./assets/seeds/Centroides Basalto.csv",
            "classification": "basalto",
        },
    ]

    allSeeds = []

    for options in optionsArray:

        print('reading segmented DEM file "{}"...'.format(options["segmentedFileName"]))
        segmentedMap = DEM(options["segmentedFileName"])
        segmentedHeightMap = segmentedMap.getNumpyHeightMap()

        print('reading seeds file "{}"...'.format(options["seedsFileName"]))
        seeds = pd.read_csv(options["seedsFileName"], ",")

        oldCols = seeds.values.shape[1]
        newCols = oldCols + 2
        rows = seeds.values.shape[0]

        newSeeds = np.empty((rows, newCols), dtype=object)
        newSeeds[:,:oldCols] = seeds.values
        newSeeds[:,-1] = options["classification"]

        print("generating histograms...")
        for seedIndex in range(rows):
            seedName = seeds.values[seedIndex, 1]
            seedDegreesX = seeds.values[seedIndex, 3]
            seedDegreesY = seeds.values[seedIndex, 4]
            seedX, seedY = segmentedMap.convertDegreesToPixels(
                seedDegreesX, seedDegreesY
            )

            radius = getSimpleRadius(segmentedHeightMap, (seedY, seedX))
            # # SIMPLE INNER SQUARE
            # radius = round(np.floor(radius* np.sin(np.pi / 5)- 1))

            # ADD HERE
            newSeeds[seedIndex, -2] = radius
            newSeeds[seedIndex, 3] = seedX
            newSeeds[seedIndex, 4] = seedY

        # SAVE HERE
        allSeeds.append(newSeeds)

        if segmentedMap:
            segmentedMap.clear()

    # make header
    header = [a for a in seeds.columns.values]
    header.append("radius_px")
    header.append("class")

    # make body
    body = np.vstack((allSeeds[0], allSeeds[1]))

    # make dataframe
    allSeedsDF = pd.DataFrame(body)

    # save
    allSeedsDF.to_csv(
        outputSeedsFileName,
        header=header,
        index=False,
    )


if __name__ == "__main__":
    main()
