import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


""" Read the data """


def read_file(file):
    with open(file, encoding='utf-16') as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row.rstrip())
        contents.pop(0)
        potentialAg = []
        potentialSHE = []
        current = []
        for line in contents:
            potentialAg.append(float(line.split('\t')[0]))
            # potentialAg.append(float(line.split(',')[4]))
            current.append(float(line.split('\t')[1]))
            # current.append(float(line.split(',')[5]))
        for value in potentialAg:
            potentialSHE.append(value+0.209)
        return potentialSHE, current


""" Create a data frame """


def create_df(potential, current):
    dict = {'Potential': potential, 'Current': current}
    df = pd.DataFrame(dict)
    return df


""" Plot the data """


def linechart(df, title):
    x = df["Potential"]
    y = df["Current"]

    plt.plot(x, y, color='black')
    plt.xlabel('Potential vs SHE (V)', size=12)
    plt.ylabel('Current intensity (nA)', size=12)
    plt.title(title)
    plt.show()


def linechart_difference(df, title):
    x = df["Potential"]
    y = df["Difference"]

    plt.plot(x, y, color='black')
    plt.xlabel('Potential vs SHE (V)', size=12)
    plt.ylabel('Difference in current intensity (nA)', size=12)
    plt.title(title, pad='25')
    plt.show()


def get_data(file):
    a, b = read_file(file)
    data = create_df(a, b)
    return data


def analyse_CV(file, title):
    a, b = read_file(file)
    data = create_df(a, b)
    linechart(data, title)
    return data


def analyse_difference(file1, file2, title):
    a, b, = read_file(file1)
    c, d = read_file(file2)
    dict = {'Potential': a, 'Current1': b, 'Current2': d}
    df = pd.DataFrame(dict)
    df["Difference"] = df['Current1'] - df['Current2']
    print(df)
    linechart_difference(df, title)


def integrate(data, a, b):
    current = data["Current"].to_list()
    potential = data["Potential"].to_list()

    half_length = len(potential) // 2
    redlist = potential[:half_length]
    redlist = [round(elem, 3) for elem in redlist]
    oxlist = potential[half_length:]
    oxlist = [round(elem, 3) for elem in oxlist]

    print(current)
    print(len(current))
    print(len(current[0]))

    A = oxlist.index(a)
    B = oxlist.index(b)
    area = np.trapz(current[A:B], x=potential[A:B])
    return area

