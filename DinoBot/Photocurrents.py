import pandas as pd
from statistics import mean
from scipy import signal
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

""" Read the data """


def read_pc_data(file):
    # opens a text/ascii file and creates three lists containing
    # time, current and potential
    with open(file) as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row.rstrip())
        contents.pop(0)
        time = []
        intensity = []
        # potential = []
        for line in contents:
            time.append(float(line.split('\t')[0]))
            intensity.append(float(line.split('\t')[1]))
            # potential.append(float(line.split('\t')[2]))

        potential = [None] * len(time)
        return time, intensity, potential


""" Processing photocurrent data """


# this function defines at which times the light is on

def light_on(df, first_on, column, time_on, time_off):
    # adds a column in the data frame with 0 for light off or 1 for light on
    for row in df.itertuples():
        if (row[1]) < first_on:
            column.append(0)

        elif (row[1]) == first_on:
            column.append(1)

        elif first_on < (row[1]) < (first_on + time_on):
            column.append(1)

        elif (first_on + time_on) <= (row[1]) < (first_on + time_on + time_off):
            column.append(0)

        else:
            first_on = first_on + time_on + time_off
            column.append(1)


def create_pc_df(time, intensity, potential, first_on, time_on, time_off):
    # creates a data frame by compiling the lists
    dict = {'Time (s)': time, 'Intensity (nA)': intensity, 'Potential (V)': potential}
    df = pd.DataFrame(dict)
    light = []
    light_on(df, first_on, light, time_on, time_off)
    df["Light on?"] = light
    return df


def identify_change(df, list_of_changes):
    # first value is always 0
    previous = 0
    for row in df.itertuples():
        # reads the light on/off column and identifies when light is switched
        if (row[4]) != previous:
            list_of_changes.append(row[1])
            previous = row[4]


def average_current_plateaus(list_of_changes, df):
    # averages on and off currents
    averages = []
    values_to_be_averaged = []
    print("The light switches at: ", list_of_changes)
    # takes 5s before a light switch and averages the values
    for l in list_of_changes:
        for row in df.itertuples():
            if l-5 < row[1] < l:
                values_to_be_averaged.append(row[2])
                mean_values = mean(values_to_be_averaged)
        averages.append(mean_values)
        values_to_be_averaged = []
    return averages


def calculate_current_densities(list_of_averages):
    photocurrent_densities = []
    # excludes the first photo-current as it is often under the influence of the Cottrell curve
    for p in list_of_averages:
        if (list_of_averages.index(p) % 2) != 0 and list_of_averages.index(p) > 0:
            index_light = list_of_averages.index(p)
            index_dark = list_of_averages.index(p)-1
            pc = list_of_averages[index_light]-list_of_averages[index_dark]
            photocurrent_densities.append(pc)
    print("Densities non-normalised: ", photocurrent_densities)
    return photocurrent_densities


def normalise_by_cm2(photocurrent_densities):
    # averages photocurrents per cm^2
    photocurrent_densities = [x / 0.785 for x in photocurrent_densities]
    return photocurrent_densities


def normalise_by_chla(photocurrent_densities, chl_a):
    # averages photocurrents per chla concentration in mmol
    photocurrent_densities = [x / chl_a for x in photocurrent_densities]
    return photocurrent_densities


def normalise_by_cell_count(photocurrent_densities, cell_count):
    # averages photocurrents per cell
    photocurrent_densities = [x / cell_count for x in photocurrent_densities]
    # then convert nA to fA as otherwise value will be too small
    photocurrent_densities = [x*(10**6) for x in photocurrent_densities]
    return photocurrent_densities


def analyse_photocurrents(name, first_on, mol_chla, time_on, time_off):
    # create a data frame from the text filename
    t, i, p = read_pc_data(name)
    data = create_pc_df(t, i, p, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    current_densities_list = calculate_current_densities(steady_states_list)

    # normalise by area and chla content
    densities_normalised_by_area = normalise_by_cm2(current_densities_list)
    densities_normalised_by_chla = normalise_by_chla(densities_normalised_by_area, mol_chla)
    densities_normalised_by_chla_formatted = [round(elem, 2) for elem in densities_normalised_by_chla]

    # Output data
    print("Densities normalised by area and chla: ", densities_normalised_by_chla_formatted)
    return data, list_of_changes, densities_normalised_by_chla_formatted


def analyse_photocurrents_per_cell(name, first_on, cell_count, time_on, time_off):
    # create a data frame from the text filename
    t, i, p = read_pc_data(name)
    data = create_pc_df(t, i, p, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    current_densities_list = calculate_current_densities(steady_states_list)

    # normalise by area and chla content
    densities_normalised_by_area = normalise_by_cm2(current_densities_list)
    densities_normalised_by_cell = normalise_by_cell_count(densities_normalised_by_area, cell_count)
    densities_normalised_by_cell_formatted = [round(elem, 2) for elem in densities_normalised_by_cell]

    # Output data
    print("Densities normalised by area and cell: ", densities_normalised_by_cell_formatted)
    return data, list_of_changes, densities_normalised_by_cell_formatted


def analyse_photocurrents_blank(name, first_on, time_on, time_off):
    # create a data frame from the text filename
    t, i, p = read_pc_data(name)
    data = create_pc_df(t, i, p, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    current_densities_list = calculate_current_densities(steady_states_list)

    # normalise by area and chla content
    densities_normalised_by_area = normalise_by_cm2(current_densities_list)
    densities_normalised_by_area_formatted = [round(elem, 2) for elem in densities_normalised_by_area]

    # Output data
    print("Densities normalised by area and chla: ", densities_normalised_by_area_formatted)
    return data, list_of_changes, densities_normalised_by_area_formatted


def analyse_photocurrents_raw(name, first_on, time_on, time_off):
    # create a data frame from the text filename
    t, i, p = read_pc_data(name)
    data = create_pc_df(t, i, p, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    current_densities_list = calculate_current_densities(steady_states_list)

    # don't normalise by area or chla content
    densities = [round(elem, 2) for elem in current_densities_list]

    # Output data
    print("Densities normalised by area and chla: ", densities)
    return data, list_of_changes, densities


def analyse_pc_dark_currents(name, first_on, time_on, time_off):
    # create a data frame from the text filename
    t, i, p = read_pc_data(name)
    data = create_pc_df(t, i, p, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    # print("steady states", steady_states_list)

    # get a half of those values
    dark_list = [steady_states_list[index] for index in range(0, len(steady_states_list), 2)]
    print(dark_list)

    # print('dark:', dark_list)
    # dark_list.pop(0)

    return data, list_of_changes, dark_list


def analyse_pc_dark_currents_per_cell(name, first_on, cell_count, time_on, time_off):
    # create a data frame from the text filename
    t, i, p = read_pc_data(name)
    data = create_pc_df(t, i, p, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    # print("steady states", steady_states_list)

    # get a half of those values
    dark_list = [steady_states_list[index] for index in range(0, len(steady_states_list), 2)]
    current_normalised_by_cell = normalise_by_cell_count(dark_list, cell_count)
    current_normalised_by_cell_formatted = [round(elem, 2) for elem in current_normalised_by_cell]

    # print('dark:', dark_list)
    # dark_list.pop(0)

    return data, list_of_changes, current_normalised_by_cell_formatted


def average_photocurrent_densities(list_of_densities, start, finish):
    subset_of_list = list_of_densities[start:finish]
    mean_density = round(mean(subset_of_list), 2)
    print("Average photocurrent density over range: ", mean_density, "nA cm^-2 (nmol chl_a)^-1")


""" Data processing for averaging photocurrents """


def split_up_photocurrents(data, list_of_changes, first_on):
    previous = first_on
    photocurrent_list = []
    for l in list_of_changes[::2]:
        # print("l is:", l)
        photocurrent = data[(previous <= data["Time (s)"]) & (data["Time (s)"] < l)]
        previous = l
        photocurrent_list.append(photocurrent)
    # print(photocurrent_list)
    return photocurrent_list


def reindex_photocurrents(photocurrent_list, list_of_changes):
    indexed_pl = []
    loc = list_of_changes[::2]

    for p, l in zip(photocurrent_list, loc):
        p_df = pd.DataFrame(p)
        p_df.reset_index(inplace=True, drop=True)
        # print(p_df)
        # print("remove: ", l)
        # not sure where the 150 comes from, maybe 1 cycle removed too much?
        p_df["Time (s)"] -= (l-150)
        indexed_pl.append(p)
    # print("new list:", indexed_pl)
    return indexed_pl


def average_photocurrents(indexed_pl, offset):
    # reindexed photocurrent list has all the photocurrents, even the empty ones (not the studied step).
    # therefore, offset = the number of the photocurrent we start from. And offset + 1 usually the one where potential
    # is changed.
    pd.set_option('display.max_columns', None)
    base_df = pd.DataFrame(indexed_pl[offset+1])
    base_df = base_df.drop(["Intensity (nA)"], axis=1)

    indexed_pl.pop(1+offset)

    for p in indexed_pl:
        p_df = pd.DataFrame(p)
        intensity = p_df["Intensity (nA)"]
        base_df = pd.concat([base_df, intensity], axis=1)

    filter_col = [col for col in base_df if col.startswith('Int')]
    base_df["average_pc"] = base_df[filter_col].mean(axis=1)
    base_df["std"] = base_df[filter_col].std(axis=1)
    base_df["ymin"] = base_df["average_pc"] - base_df["std"]
    base_df["ymax"] = base_df["average_pc"] + base_df["std"]
    # print(base_df)
    return base_df


def average_photocurrents_trimmed(indexed_pl, offset, outlier):
    pd.set_option('display.max_columns', None)
    base_df = pd.DataFrame(indexed_pl[offset+1])
    base_df = base_df.drop(["Intensity (nA)"], axis=1)
    indexed_pl.pop(outlier)
    indexed_pl.pop(0+offset)

    for p in indexed_pl:
        p_df = pd.DataFrame(p)
        intensity = p_df["Intensity (nA)"]
        base_df = pd.concat([base_df, intensity], axis=1)

    filter_col = [col for col in base_df if col.startswith('Int')]
    base_df["average_pc"] = base_df[filter_col].mean(axis=1)
    base_df["std"] = base_df[filter_col].std(axis=1)
    base_df["ymin"] = base_df["average_pc"] - base_df["std"]
    base_df["ymax"] = base_df["average_pc"] + base_df["std"]
    # print(base_df)
    return base_df


def get_average_df(data, loc, first_on, offset):
    plist0 = split_up_photocurrents(data, loc, first_on)
    ri_plist = reindex_photocurrents(plist0, loc)
    base_df = average_photocurrents(ri_plist, offset)
    # print("BASE DF", base_df)
    return base_df


def get_average_df_trimmed(data, loc, first_on, offset, outlier):
    plist0 = split_up_photocurrents(data, loc, first_on)
    ri_plist = reindex_photocurrents(plist0, loc)
    base_df = average_photocurrents_trimmed(ri_plist, offset, outlier)
    print("TRIMMED BASE DF", base_df)
    return base_df


def average_replicate_shapes(df1, df2, df3):
    df1.rename(columns={'Time (s)': 'Time'}, inplace=True)
    df1.rename(columns={'average_pc': 'average_pc1'}, inplace=True)
    df1.rename(columns={'std': 'std1'}, inplace=True)
    df2.rename(columns={'average_pc': 'average_pc2'}, inplace=True)
    df2.rename(columns={'std': 'std2'}, inplace=True)
    df3.rename(columns={'average_pc': 'average_pc3'}, inplace=True)
    df3.rename(columns={'std': 'std3'}, inplace=True)
    master_df = pd.concat([df1, df2, df3], axis=1)
    filter_col = [col for col in master_df if col.startswith('average')]
    master_df["all_averages"] = master_df[filter_col].mean(axis=1)
    # average of stds is: sqrt((std1**2 + std2**2 + std3**2)/k) where k is number of groups
    master_df["Total_error"] = np.sqrt((master_df["std1"]**2 + master_df["std1"]**2 + master_df["std1"]**2)/3)
    master_df["all_ymin"] = master_df["all_averages"] - master_df["Total_error"]
    master_df["all_ymax"] = master_df["all_averages"] + master_df["Total_error"]
    print(" MASTER DF \n", master_df)
    return master_df


def average_whole_replicate_shapes(df1, df2, df3):
    df1.rename(columns={'Time (s)': 'Time'}, inplace=True)
    df1.rename(columns={'average_pc': 'average_pc1'}, inplace=True)
    df2.rename(columns={'average_pc': 'average_pc2'}, inplace=True)
    df3.rename(columns={'average_pc': 'average_pc3'}, inplace=True)
    base_df = pd.concat([df1, df2, df3], axis=1)
    filter_col = [col for col in base_df if col.startswith('Int')]
    base_df["all_averages"] = base_df[filter_col].mean(axis=1)
    base_df["std"] = base_df[filter_col].std(axis=1)
    base_df["all_ymin"] = base_df["all_averages"] - base_df["std"]
    base_df["all_ymax"] = base_df["all_averages"] + base_df["std"]
    # print(" MASTER DF \n", master_df)
    return base_df


""" Additional functions for data analysis """


# this function is useful when the data is very noisy

def clean_data(data):
    d = data["Intensity (nA)"]

    # # max freq = sampling freq/2
    # f = np.linspace(0, 5, len(d))
    # spectrum = np.fft.fft(data["Intensity (nA)"])
    #
    # # plot the Fourier transform, the high frequency peaks can usually be eliminated
    # plt.figure()
    # plt.plot(f, np.abs(spectrum))
    # plt.show()

    # Apply a butterworth filter to clean the data
    # signal(order of filter/sharpness, cuttoff freq at which power is half, type)
    sos = signal.butter(1, 0.2, 'lp', fs=10, output='sos')
    clean_signal = signal.sosfilt(sos, data["Intensity (nA)"])
    print(clean_signal)

    # returns the filtered data, you can plot this
    return clean_signal


# baselining

def baseline(dataset):
    # measure dark current
    darkdata = dataset[dataset["Light on?"] == 0]
    # print(darkdata)
    # plt.plot(darkdata["Time (s)"], darkdata["Intensity (nA)"])
    # plt.show()
    darkmean = darkdata.loc[:, 'Intensity (nA)'].mean()
    # print(darkmean)
    dataset["Intensity (nA)"] -= darkmean
    # print(dataset)
    return dataset


def baseline_with_linear_model(data):
    # model data to straight line
    x = data["Time"].to_numpy()
    x = x.reshape(-1, 1)
    y = data["O2"].to_numpy()
    y = y.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    # calculate dataset for that model
    x_new = np.arange(0, len(x), 1).reshape((-1, 1))
    y_new = model.predict(x_new)
    plt.plot(x_new, y_new, color='k')
    plt.plot(x, y, color='blue')
    plt.show()

    # remove dataset from first dataset
    data["fit"] = y_new
    data["O2"] = data["O2"] - data["fit"]
    plt.plot(data["O2"], data["Time"], color='k')
    plt.plot(x, y, color='blue')
    plt.show()
    return data


def baseline_with_poly_model(data, degree):
    x = data["Time"].to_numpy()
    x = x.reshape(-1, 1)
    y = data["O2"].to_numpy()
    y = y.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(poly_features, y)
    y_predicted = model.predict(poly_features)

    # plt.plot(x, y_predicted, color='k')
    # plt.plot(x, y, color='blue')
    # plt.show()

    data["fit"] = y_predicted
    data["O2"] = data["O2"] - data["fit"]
    # plt.plot(data["Time"], data["O2"], color='k')
    # plt.plot(x, y, color='blue')
    # plt.show()
    return data


def exp_function(x, a, b, c):
    return a * np.exp(b * x) + c


def exp_function2(x, a, b):
    return a * np.exp(-b * x)


def cottrell(x, n, cj, D):
    return (n*96485*0.785*cj*np.sqrt(D))/(np.sqrt(np.pi*x))


def integrate_dark_current(data, loc):
    dips = []
    dark_data = []
    data = baseline(data)
    for l in loc:
        if loc.index(l) % 2 != 0:
            l2 = loc[loc.index(l) + 1]
            print("l and l2", l, l2)
            darkfilter = data[(data["Time (s)"] > l) & (data["Time (s)"] < l2)]
            dark_data.append(darkfilter)
    print(dark_data)
    for d in dark_data:
        area = - np.trapz(d["Intensity (nA)"], dx=0.1)
        dips.append(area)
    return dips


def integrate_dark_dip(data, loc):
    dips = []
    dark_data = []
    for l in loc:
        if loc.index(l) % 2 != 0:
            if loc.index(l) < 11:
                l2 = loc[loc.index(l) + 1]
                print("l and l2", l, l2)
                # dark dip usually lasts 20 s
                darkfilter = data[(data["Time (s)"] > l) & (data["Time (s)"] < l + 20)]
                dark_data.append(darkfilter)
    print(dark_data)
    for d in dark_data:
        area = np.abs(np.trapz(d["Intensity (nA)"], dx=0.1))
        # area = -(np.trapz(d["Intensity (nA)"], dx=0.1))
        dips.append(area)
    return dips

