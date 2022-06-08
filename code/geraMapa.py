from email import header
from .DEM import DEM
from joblib import load, dump
import cv2
import numpy as np
import pandas as pd

def laplacianCv2(mat):
    filteredMat = cv2.Laplacian(mat, cv2.CV_16S, ksize=3)
    return filteredMat

def robertsMag(mat, window=2):
    half = int(window / 2)
    ones = np.ones((half, half))
    mat = np.array(mat, dtype=np.float32)
    Gx = np.zeros((window, window))
    Gx[:half, half:] = ones
    Gx[half:, :half] = -ones
    Gy = np.zeros((window, window))
    Gy[:half, :half] = ones
    Gy[half:, half:] = -ones
    RobertsX = cv2.filter2D(mat, cv2.CV_32F, Gx)
    RobertsY = cv2.filter2D(mat, cv2.CV_32F, Gy)
    return np.sqrt(np.add(np.square(RobertsX), np.square(RobertsY)))

def geraMapa():
    windowSize = 16
    modelName = "classifier{}_2.joblib".format(windowSize)

    print('abrindo arquivo...')
    dem = DEM("./assets/dems/S028-030_W053-055.tif")
    map = dem.getNumpyHeightMap()

    print('extraindo caracterísicas do mapa...')
    mapSideSize = map.shape[0]
    list = []
    for startX in range(0, mapSideSize, windowSize):
        for startY in range(0, mapSideSize, windowSize):
            area = map[startY:startY+windowSize, startX:startX+windowSize]
            list.append({
                "mean": np.mean(area),
                "std": np.std(area),
                "roberts_mag_mean": np.mean(robertsMag(area)),
                "laplacian_cv2_std": np.std(laplacianCv2(area)),
            })
        #     break
        # break
    dem.clear()
    x_map = pd.DataFrame(list)

    print('classificando áreas...')
    modelFile = "./assets/classes/new/exphist/{}/{}".format(windowSize, modelName)
    model = load(modelFile)
    y_map = pd.DataFrame(model.predict(x_map), columns=["classified_as"])
    print(x_map)
    print(y_map)

    print('salvando resultados...')
    x_map.to_csv("./resultados/geraMapa/{}_x.csv".format(windowSize), index=None)
    y_map.to_csv("./resultados/geraMapa/{}_y.csv".format(windowSize), index=None)
    dump(model, "./resultados/geraMapa/{}_model.joblib".format(windowSize))

    print('gerando novo mapa...')
    minimapSideSize = round(np.floor(mapSideSize / windowSize))
    minimap = np.zeros([minimapSideSize, minimapSideSize])
    y_line = 0
    test = {}
    for vi in range(0,minimapSideSize):
        for hi in range(0,minimapSideSize):
            y = y_map.values[y_line,0]
            if y == "arenito":
                minimap[vi,hi] = 255
            if y not in test:
                test[y] = 1
            else:
                test[y] += 1
            y_line+=1
    print(test)
    cv2.imwrite("./resultados/geraMapa/{}_map.tif".format(windowSize), minimap)

    print('arquivos resultantes salvos em "/resultados/geraMapa/"')