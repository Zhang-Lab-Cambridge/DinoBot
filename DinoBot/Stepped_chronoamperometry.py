import numpy as np
import pandas as pd
from statistics import mean


""" Read the data """


def read_sc_data(file):
    with open(file) as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row.rstrip())
        contents.pop(0)
        time = []
        intensity = []
        for line in contents:
            time.append(float(line.split('\t')[0]))
            intensity.append(float(line.split('\t')[1]))
        return time, intensity


""" Process the data """


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


def create_sc_df(time, intensity, first_on, time_on, time_off):
    # creates a data frame by compiling the lists
    dict = {'Time (s)': time, 'Intensity (nA)': intensity}
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
        if (row[3]) != previous:
            list_of_changes.append(row[1])
            previous = row[3]


def average_current_plateaus(list_of_changes, df):
    # averages on and off currents
    averages = []
    values_to_be_averaged = []
    print("The light switches at: ", list_of_changes)
    # takes 5s before a light switch and averages the values
    mean_values = 0
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
    # excludes the first photo-current as it is often under the influence of the Cottrell
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


def average_photocurrent_densities(list_of_densities, start, finish):
    subset_of_list = list_of_densities[start:finish]
    mean_density = round(mean(subset_of_list), 2)
    print("Average photocurrent density over range: ", mean_density, "nA cm^-2 (nmol chl_a)^-1")


def analyse_stepped_chrono(name, first_on, mol_chla, time_on, time_off):
    # create a data frame from the text filename
    t, i = read_sc_data(name)
    data = create_sc_df(t, i, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    current_densities_list = calculate_current_densities(steady_states_list)

    # normalise by area and chla content
    densities_normalised_by_area = normalise_by_cm2(current_densities_list)
    print("By area:", densities_normalised_by_area)
    densities_normalised_by_chla = normalise_by_chla(densities_normalised_by_area, mol_chla)
    densities_normalised_by_chla_formatted = [round(elem, 2) for elem in densities_normalised_by_chla]

    # Output data
    print("Densities normalised by area and chla: ", densities_normalised_by_chla_formatted)
    return data, list_of_changes, densities_normalised_by_chla_formatted


def analyse_stepped_chrono_per_cell(name, first_on, cell_count, time_on, time_off):
    # create a data frame from the text filename
    t, i = read_sc_data(name)
    data = create_sc_df(t, i, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    current_densities_list = calculate_current_densities(steady_states_list)

    # normalise by area and chla content
    densities_normalised_by_area = normalise_by_cm2(current_densities_list)
    print("By area:", densities_normalised_by_area)
    densities_normalised_by_cell = normalise_by_cell_count(densities_normalised_by_area, cell_count)
    densities_normalised_by_cell_formatted = [round(elem, 2) for elem in densities_normalised_by_cell]

    # Output data
    print("Densities normalised by area and chla: ", densities_normalised_by_cell_formatted)
    return data, list_of_changes, densities_normalised_by_cell_formatted


def analyse_stepped_chrono_blank(name, first_on, time_on, time_off):
    # create a data frame from the text filename
    t, i = read_sc_data(name)
    data = create_sc_df(t, i, first_on, time_on, time_off)

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
    print("Densities normalised by area ", densities_normalised_by_area_formatted)
    return data, list_of_changes, densities_normalised_by_area_formatted


def analyse_sc_dark_currents(name, first_on, time_on, time_off):
    # create a data frame from the text filename
    t, i = read_sc_data(name)
    data = create_sc_df(t, i, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    # get a list of the current steady states
    steady_states_list = average_current_plateaus(list_of_changes, data)
    # print("steady states", steady_states_list)

    # get a half of those values
    dark_list = [steady_states_list[index] for index in range(0, len(steady_states_list), 2)]
    # print('dark:', dark_list)
    dark_list.pop(0)

    return data, list_of_changes, dark_list


# this function groups together the replicates for making box plots

def sc_replicate_to_3(list_of_densities):
    del list_of_densities[0::4]
    # print(list_of_densities)
    list_of_densities = [list_of_densities[x:x + 3] for x in range(0, len(list_of_densities), 3)]
    print(list_of_densities)
    return list_of_densities


def group_by_3(list_of_densities):
    list_of_densities = [list_of_densities[x:x + 3] for x in range(0, len(list_of_densities), 3)]
    # print(list_of_densities)
    return list_of_densities


def relative_current(densities):
    # print("densities: ", densities)
    relative_densities = []
    base_tuplet = densities[0]
    # print("base_tuplet", base_tuplet)
    base = abs(np.mean(base_tuplet))
    # print("base", base)
    for l in densities:
        # print("l", l)
        for d in l:
            current = abs(d - base)
            relative_densities.append(current)
    relative_densities = group_by_3(relative_densities)
    print(relative_densities)
    return relative_densities


def split_into_potentials(data, loc, lop):
    # print("\nloc:\n", loc)
    # to go from time to index: - 0.1 * 10
    loc = loc[::8]
    # print(loc)
    steps = []
    for l in loc:
        steps.append(data[int((l-0.1) * 10):int(((l+600)-0.1) * 10)])
    # print("STEPS", steps)
    return steps
