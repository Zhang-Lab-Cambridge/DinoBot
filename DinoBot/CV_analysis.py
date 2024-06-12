import matplotlib.pyplot as plt
import pandas as pd


""" Read the data """


def read_file(file):
    with open(file) as t:
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
            # potentialAg.append(float(line.split(',')[0]))
            current.append(float(line.split('\t')[1]))
            # current.append(float(line.split(',')[1]))
        for value in potentialAg:
            potentialSHE.append(value+0.209)
        return potentialSHE, current


""" Create a data frame """


def create_df(potential, current):
    dict = {'Potential (V)': potential, 'Current (nA)': current}
    df = pd.DataFrame(dict)
    return df


""" Plot the data """


def linechart(df, title):
    x = df["Potential (V)"]
    y = df["Current (nA)"]

    plt.plot(x, y, color='black')
    plt.xlabel('Potential vs SHE (V)', size=12)
    plt.ylabel('Current intensity (nA)', size=12)
    plt.title(title)
    plt.show()


def linechart_difference(df, title):
    x = df["Potential (V)"]
    y = df["Difference (nA)"]

    plt.plot(x, y, color='black')
    plt.xlabel('Potential vs SHE (V)', size=12)
    plt.ylabel('Difference in current intensity (nA)', size=12)
    plt.title(title, pad='25')
    plt.show()


def analyse_CV(file, title):
    a, b = read_file(file)
    data = create_df(a, b)
    linechart(data, title)
    return data


def analyse_difference(file1, file2, title):
    a, b, = read_file(file1)
    c, d = read_file(file2)
    dict = {'Potential (V)': a, 'Current1 (nA)': b, 'Current2 (nA)': d}
    df = pd.DataFrame(dict)
    df["Difference (nA)"] = df['Current1 (nA)'] - df['Current2 (nA)']
    print(df)
    linechart_difference(df, title)

