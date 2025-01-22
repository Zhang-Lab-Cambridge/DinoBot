import csv
import codecs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_SECM_data(file):
    contents = csv.reader(codecs.open(file))
    potential = []
    current = []
    for line in contents:
        print(line)
        potential.append(float(line[0]))
        current.append(float(line[1]))
    print(potential)
    print(current)
    potential = [x + 0.209 for x in potential]  # converts to SHE
    return potential, current


def read_SECM_data_bin(file):
    with open(file, "rb") as file:
        data = file.read()
        print(data)
    with open("out.txt", "wb") as f:
        f.write(data)


def create_SECM_CV_df(potential, current):
    dict = {'Potential': potential, 'Current': current}
    df = pd.DataFrame(dict)
    return df


def linechart(df, title):
    x = df["Potential"]
    y = df["Current"]

    plt.plot(x, y, color='black')
    plt.xlabel('Potential vs SHE (V)', size=12)
    plt.ylabel('Current intensity (nA)', size=12)
    plt.title(title)
    plt.show()


def analyse_SECM_CV(file, title):
    a, b = read_SECM_data(file)
    data = create_SECM_CV_df(a, b)
    linechart(data, title)
    return data


def norm(current):
    currentbycm2 = current/(np.pi*(7.5e-4)**2) # radius of the Pt tip
    currentbycm2mA = currentbycm2 * 1000
    return currentbycm2mA
