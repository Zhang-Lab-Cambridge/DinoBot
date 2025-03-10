import pandas as pd
import numpy as np
from Photocurrents import baseline_with_linear_model
from Photocurrents import baseline_with_poly_model


def read_o2_data(file):
    # time interval is 1s
    with open(file, mode="r", encoding="utf-8") as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row.rstrip())
        # contents.pop(0)
        time = []
        conc_O2 = []
        for line in contents:
            time.append((line.split('\t')[0]))
            conc_O2.append(float(line.split('\t')[1]))
        return time, conc_O2


def read_temp_data(file):
    # time interval is 0.5s!
    with open(file, mode="r", encoding="utf-8") as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row.rstrip())
        # contents.pop(0)
        time = []
        temp = []
        for line in contents:
            time.append((line.split('\t')[0]))
            temp.append(float(line.split('\t')[1]))
        return time, temp


def replace_time(datetime):
    l = len(datetime)
    newtime = np.arange(0, l+1, 1) # should be 1 because interval = 1 second
    return newtime


def intersperse(list, item):
    newlist = [item] * (len(list) * 2)
    newlist[0::2] = list
    newlist.append(None)
    return newlist


def create_df(time, O2, temp, first_on, time_on, time_off):
    dict = {"Time": time, "O2": O2, "Temperature": temp}
    df = pd.DataFrame(dict)
    light = []
    light_on(df, first_on, light, time_on, time_off)
    df["Light on?"] = light
    return df


def create_df_no_temp(time, O2, first_on, time_on, time_off):
    dict = {"Time": time, "O2": O2}
    df = pd.DataFrame(dict)
    light = []
    light_on(df, first_on, light, time_on, time_off)
    df["Light on?"] = light
    return df


def read_just_o2_data(file, first_on, time_on, time_off):
    time, conc_O2 = read_o2_data(file)
    # check if starts on second or half second
    # conc_O2 = intersperse(conc_O2, None)
    # print(conc_O2)
    df = create_df_no_temp(time, conc_O2, first_on, time_on, time_off)
    return df


def read_all_o2_data(file, file2, first_on, time_on, time_off):
    time, conc_O2 = read_o2_data(file)
    # check if starts on second or half second
    # conc_O2 = intersperse(conc_O2, None)
    # print(conc_O2)
    time2, temp = read_temp_data(file2)
    if len(temp) > len(conc_O2):
        temp = temp[0:len(conc_O2)]
    elif len(temp) < len(conc_O2):
        conc_O2 = conc_O2[0:len(temp)]
    # print(time[0])
    # print(time2[0])
    print('l time', len(time))
    print('l time2', len(time2))
    # print('l conc_O2', len(conc_O2))
    # print('l temp', len(temp))
    if time[0] == time2[0]:
        newtime = replace_time(time)
        # print(newtime)
        # print('l new time:', len(newtime))
        newtime = newtime[0:len(conc_O2)]
        df = create_df(newtime, conc_O2, temp, first_on, time_on, time_off)
        return df
    else:
        print("times not matching!")


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


def identify_change(df, list_of_changes):
    # first value is always 0
    previous = 0
    for row in df.itertuples():
        # reads the light on/off column and identifies when light is switched
        if (row[4]) != previous:
            list_of_changes.append(row[1])
            previous = row[4]


def get_gradients(df, list_of_changes):
    models = []
    l_0 = 0
    for l in list_of_changes:
        # print('l0 and l:', l_0, l)
        filter = df.loc[(l_0 < df["Time"]) & (df["Time"] < l) & (df["O2"].notna())]
        # print(filter)
        p = np.polyfit(filter["Time"], filter["O2"], 1)
        model = np.poly1d(p)
        models.append(model.c[0])
        l_0 = l
    return models


def analyse_o2_evolution(o2_file, temp_file, first_on, time_on, time_off):
    # create a data frame from the text filename
    data = read_all_o2_data(o2_file, temp_file, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    gradients = get_gradients(data, list_of_changes)
    gradients = gradients[1::2]
    # print(data)
    return data, list_of_changes, gradients


def analyse_o2_evolution_fitted(o2_file, temp_file, first_on, time_on, time_off, fit, degree):
    # create a data frame from the text filename
    data = read_all_o2_data(o2_file, temp_file, first_on, time_on, time_off)
    if fit == 'linear':
        data = baseline_with_linear_model(data)
    elif fit == 'poly':
        data = baseline_with_poly_model(data, degree)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    gradients = get_gradients(data, list_of_changes)
    gradients = gradients[1::2]
    # print(data)
    return data, list_of_changes, gradients


def analyse_just_o2(o2_file, temp_file, first_on, time_on, time_off):
    data = read_just_o2_data(o2_file, first_on, time_on, time_off)

    # identify the changes in light switching
    list_of_changes = []
    identify_change(data, list_of_changes)

    gradients = get_gradients(data, list_of_changes)
    gradients = gradients[1::2]
    # print(data)
    return data, list_of_changes, gradients


def get_moles_photocurrent(average_photocurrent_nA, error_photocurrent, time_s):
    charge = average_photocurrent_nA * 1e-9 * time_s
    mole_e = charge / 96486
    error_e = (np.sqrt((error_photocurrent/average_photocurrent_nA)**2)) * mole_e
    # print(mole_e, error_e)
    return mole_e, error_e


def get_moles_made_by_o2(average_O2_umol_L, error_O2, volume_L, cell_count):
    scaling_factor = 3e6 / cell_count
    mole_o2 = (average_O2_umol_L * 1e-6 * volume_L) * scaling_factor
    moles_e_from_O2 = mole_o2 * 4
    error_e_from_O2 = (np.sqrt((error_O2/average_O2_umol_L)**2)) * moles_e_from_O2
    # print(moles_e_from_O2, error_e_from_O2)
    return moles_e_from_O2, error_e_from_O2


def get_percentage_exported(mole_exported, error_exported, moles_made, error_made):
    percentage_exported = (mole_exported / moles_made) * 100
    error_e_exported = np.sqrt((error_exported / mole_exported)**2 + (error_made / moles_made) ** 2) * percentage_exported
    # print(percentage_exported, error_e_exported)
    return percentage_exported, error_e_exported


def get_electrons_exported(average_photocurrent_nA, error_photocurrent, time_s, average_O2_umol_L, error_O2, volume, cell_count):
    mole_exported, err_exported = get_moles_photocurrent(average_photocurrent_nA, error_photocurrent,time_s)
    # print("moles exported:", mole_exported)
    mole_made, err_made = get_moles_made_by_o2(average_O2_umol_L, error_O2, volume, cell_count)
    # print("moles made:", mole_made)
    percentage_exported, err_percentage = get_percentage_exported(mole_exported, err_exported, mole_made, err_made)
    print("Final Percentage", percentage_exported)
    print("Final Percentage error", err_percentage)
    return percentage_exported, err_percentage


