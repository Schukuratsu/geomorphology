import numpy as np
import pandas as pd

files = [
    {"name": "./assets/histograms/Arenito(raw).csv", 'limits': [0, 1000]},
    {"name": "./assets/histograms/Arenito(laplacian4).csv",
     'limits': [-500, 500]},
    {"name": "./assets/histograms/Arenito(laplacian8).csv",
     'limits': [-500, 500]},
    {"name": "./assets/histograms/Basalto(raw).csv", 'limits': [0, 1000]},
    {"name": "./assets/histograms/Basalto(laplacian4).csv",
     'limits': [-500, 500]},
    {"name": "./assets/histograms/Basalto(laplacian8).csv",
     'limits': [-500, 500]},
]

for file in files:
    df = pd.read_csv(file,)
    values = df.values
    for i in range(values.shape[1]):
        desvpad = np.desvpad(values[:, i])
