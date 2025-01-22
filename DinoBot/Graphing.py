""" This python file contains all the functions used to represent data visually """

import matplotlib.pyplot as plt
from scipy.signal import lfilter
from sklearn.linear_model import LinearRegression


""" Commonly used units """

# nA cm$^{-2}$ (nmol chl$_a$)$^{-1}$
# fA cm$^{-2}$ cell$^{-1}$

""" Basic functions """


def draw_rectangle(x, y, width, height, colour):
    x = [x, x+width, x+width, x]
    y = [y, y, y+height, y+height]
    plt.fill(x, y, alpha=0.1, color=colour)


def draw_rectangle_on_plot(list_of_changes, yu, yl):
    last = list_of_changes[-1]
    list_of_changes.append(last + 120)
    draw_rectangle(-10, yl, 10, (yu - yl) + 30, 'grey')
    for d in list_of_changes:
        if list_of_changes.index(d) == 0:
            draw_rectangle(0, yl, d, (yu - yl) + 30, 'grey')
        elif list_of_changes.index(d) % 2 == 0:
            c = list_of_changes[list_of_changes.index(d) - 1]
            draw_rectangle(c, yl, d - c, (yu - yl) + 30, 'grey')


def linear_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    plt.plot(x, model.predict(x), color='blue')
    plt.show()
    return model


""" Simple plot """


def scatter_chart(x, y, title, xlabel, ylabel):
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.title(title)
    plt.scatter(x, y, s=0.1, color='black')
    plt.show()


def line_chart(time_list, densities_list, title, xlabel, ylabel):
    x = time_list
    y = densities_list
    plt.scatter(x, y, color='black')
    plt.xlabel(xlabel, size=20)
    plt.ylabel(ylabel, size=20)
    plt.title(title, size=22)
    plt.xticks(fontsize=17)
    plt.xticks(fontsize=17)
    plt.tight_layout()
    plt.show()


""" Plots with light/dark """


def ld_scatter_chart(df, list_of_changes, title):
    x = df["Time (s)"]
    y = df["Intensity (nA)"]
    plt.xlabel('Time (s)', size=12)
    plt.ylabel('Current intensity (nA)', size=12)
    plt.title(title)

    p = plt.scatter(x, y, s=0.1, color='black')
    axd = p.axes
    axd.margins(x=0, y=0.01)
    yl, yu = axd.get_ylim()
    draw_rectangle_on_plot(list_of_changes, yu, yl)
    plt.show()


def ld_line_chart(df, list_of_changes, title):
    x = df["Time (s)"]
    y = df["Intensity (nA)"]

    plt.xlabel('Time (s)', size=16)
    plt.ylabel('Current (nA)', size=16)
    plt.title(title, size=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    p_chassis = plt.scatter(x, y, color='none')
    plt.plot(x, y, color='black', linewidth=1)

    axd = p_chassis.axes
    axd.margins(x=0, y=0.01)
    yl, yu = axd.get_ylim()
    draw_rectangle_on_plot(list_of_changes, yu, yl)
    plt.tight_layout()
    plt.show()


def ld_line_chart_o2(df, list_of_changes, title):
    x = df["Time"]
    y = df["O2"]

    plt.xlabel('Time (s)', size=16)
    plt.ylabel('O2 concentration (umol/L)', size=16)
    plt.title(title, size=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    p_chassis = plt.scatter(x, y, color='k')

    axd = p_chassis.axes
    axd.margins(x=0, y=0.01)
    yl, yu = axd.get_ylim()
    draw_rectangle_on_plot(list_of_changes, yu, yl)
    plt.tight_layout()
    plt.show()


def filtered_line(df_filtered, df_original, list_of_changes, title):
    x = df_original["Time (s)"]
    y = df_filtered

    plt.xlabel('Time (s)', size=22)
    plt.ylabel('Current (nA)', size=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.title(title, size=22)

    p_chassis = plt.scatter(x, y, color='none')
    plt.plot(x, df_original["Intensity (nA)"], linewidth=0.1, color='black')
    plt.plot(x, y, color='black', linewidth=1)

    axd = p_chassis.axes
    axd.margins(x=0, y=0.01)
    yl, yu = axd.get_ylim()
    draw_rectangle_on_plot(list_of_changes, yu, yl)
    plt.tight_layout()
    plt.show()


""" Box plots """


# Comparing photocurrent densities during chronoamperometry

def boxplot_pc(densities, title, xlabel, ylabel):
    plt.ylabel(ylabel, size=12)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.boxplot(densities, patch_artist=True,
                boxprops=dict(facecolor="green", color="black"), medianprops=dict(color="black"))
    plt.show()


# Comparing photocurrent densities during stepped chronoamperometry

def boxplot_sc(list_of_replicates, list_of_potentials, title, xlabel, ylabel):
    plt.xlabel(xlabel, size=14)
    plt.ylabel(ylabel, size=14)
    plt.title(title, size=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.boxplot(list_of_replicates, labels=list_of_potentials, patch_artist=True,
                boxprops=dict(facecolor="green", color="black"), medianprops=dict(color="black"))
    plt.show()


def boxplot_sc_blank(list_of_replicates, list_of_potentials, title, ylabel):
    plt.xlabel('Potential vs SHE', size=14)
    plt.ylabel(ylabel, size=12)
    plt.title(title, size=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.boxplot(list_of_replicates, labels=list_of_potentials, patch_artist=True,
                boxprops=dict(facecolor="grey", color="black"), medianprops=dict(color="black"))
    plt.show()


# Other


def plot_light_comparison(list_of_datasets, list_of_list_of_changes, list_of_labels, title, xlabel, ylabel):
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.title(title)

    list_of_colours = ['black', 'blue', 'green', 'orange', 'red', 'purple']
    for dataset in list_of_datasets:
        x = dataset["Time (s)"]
        y = dataset["Intensity (nA)"]
        p = plt.plot(x, y, color=list_of_colours[list_of_datasets.index(dataset)],
                 label=list_of_labels[list_of_datasets.index(dataset)], linewidth=1)

    axd = p.axes
    axd.legend(loc='upper right', markerscale=10)

    axd.margins(x=0, y=0.01)
    yl, yu = axd.get_ylim()
    draw_rectangle_on_plot(list_of_list_of_changes[0], yu, yl)

    plt.show()


def comparative_boxplot(list_of_variables, list_of_datasets, title, xlabel, ylabel):
    plt.xlabel(xlabel, size=12)
    plt.ylabel(ylabel, size=12)
    plt.title(title)

    colourlist = ['orangered', 'orange', 'gold', 'limegreen', 'green', 'dodgerblue', 'royalblue', 'purple']
    colourlist.reverse()

    p = plt.boxplot(list_of_datasets, labels=list_of_variables, patch_artist=True,
                    medianprops=dict(color="black"))

    for patch, color in zip(p['boxes'], colourlist):
        patch.set_facecolor(color)

    plt.tight_layout()
    plt.show()


""" Plotting photocurrent averages across multiple technical replicates """


def plotAverage(average_data, list_of_changes, title):
    x = average_data["Time (s)"]
    y = average_data["average_pc"]
    ymin = average_data["ymin"]
    ymax = average_data["ymax"]

    plt.xlabel('Time (s)', size=16)
    plt.ylabel('Current (nA)', size=16)
    plt.title(title, size=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    p_chassis = plt.scatter(x, y, color='none')
    plt.plot(x, y, color='black', linewidth=1)

    plt.fill_between(x, ymin, ymax, alpha=0.1, label='error band', color='black')

    axd = p_chassis.axes
    axd.margins(x=0, y=0.01)
    yl, yu = axd.get_ylim()
    list_of_changes = [x-30 for x in list_of_changes]
    draw_rectangle_on_plot(list_of_changes, yu, yl)

    plt.show()


def plot_reps_average(average_data, list_of_changes, title):
    x = average_data["Time"]
    y = average_data["all_averages"]
    ymin = average_data["all_ymin"]
    ymax = average_data["all_ymax"]

    plt.xlabel('Time (s)', size=16)
    plt.ylabel('Current (nA)', size=16)
    plt.title(title, size=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    p_chassis = plt.scatter(x, y, color='none')
    plt.plot(x, y, color='black', linewidth=1)

    plt.fill_between(x, ymin, ymax, alpha=0.1, label='error band', color='black')

    axd = p_chassis.axes
    axd.margins(x=0, y=0.01)
    yl, yu = axd.get_ylim()
    list_of_changes = [x-30 for x in list_of_changes]
    draw_rectangle_on_plot(list_of_changes, yu, yl)

    plt.show()


