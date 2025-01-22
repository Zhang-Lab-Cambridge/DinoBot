import numpy as np
import pandas as pd


# this function takes a list of densities and calculates
# the average and STDEV for each parameter (e.g. potential)
def pc_get_average_std(d_list, parameter):
    parameters = []
    averages = []
    stdevs = []

    parameters.append(parameter)

    a = np.average(d_list)
    averages.append(a)

    std = np.std(d_list)
    stdevs.append(std)

    dict = {"Parameters": parameters, "Average": averages, "STDEV": stdevs}
    df = pd.DataFrame(dict)
    print(df)
    return df

# the average and SEM for each parameter (e.g. potential)
def pc_get_average_sem(d_list, parameter):
    parameters = []
    averages = []
    sems = []

    parameters.append(parameter)

    a = np.average(d_list)
    averages.append(a)

    sem = (np.std(d_list)/np.sqrt(len(d_list)))
    sems.append(sem)

    dict = {"Parameters": parameters, "Average": averages, "SEM": sems}
    df = pd.DataFrame(dict)
    print(df)
    return df


# this function takes a list of densities list and calculates
# the average and STDEV for each parameter (e.g. potential)
def sc_get_average_std(d_list, p_list):
    parameters = []
    averages = []
    stdevs = []
    for d in d_list:
        # print("d is:", d)
        p = p_list[d_list.index(d)]
        parameters.append(p)
        a = np.average(d)
        averages.append(a)
        std = np.std(d)
        stdevs.append(std)
    dict = {"Parameters": parameters, "Average": averages, "STDEV": stdevs}
    df = pd.DataFrame(dict)
    print(df)
    return df


# the average and SEM for each parameter (e.g. potential)
def sc_get_average_sem(d_list, p_list):
    parameters = []
    averages = []
    sems = []
    for d in d_list:
        # print("d is:", d)
        p = p_list[d_list.index(d)]
        parameters.append(p)
        a = np.average(d)
        averages.append(a)
        sem = (np.std(d)/np.sqrt(len(d)))
        sems.append(sem)
    dict = {"Parameters": parameters, "Average": averages, "SEM": sems}
    df = pd.DataFrame(dict)
    print(df)
    return df


# this function is similar to sc_get_average but also converts
# the averages to percentages relative to the initial value and propagates the error
def get_percent_change(d_list, p_list):
    parameters = []
    averages = []
    sems = []
    for d in d_list:
        # print("d is:", d)
        p = p_list[d_list.index(d)]
        parameters.append(p)
        a = np.average(d)
        averages.append(a)
        # std = np.std(d)
        sem = (np.std(d)/np.sqrt(len(d)))
        sems.append(sem)
    dict = {"Parameters": parameters, "Average": averages, "SEM": sems}
    df = pd.DataFrame(dict)
    # print(df)
    initial_average = df["Average"].iloc[0]
    error_avi = df["SEM"].iloc[0]
    df["Pc_change"] = (df["Average"]/initial_average)*100
    df["Error_pc_change"] = (np.sqrt((df["SEM"]/df["Average"])**2
                                     + (error_avi/initial_average)**2
                                     ) * df["Pc_change"]
                             )
    # df["Error_pc_change"] = (np.sqrt((df["STDEV"]/df["Average"])**2
    #                                  ) * df["Pc_change"]
    #                          )

    print(df)
    return df


# this function takes the average values from one replicate and averages them with
# the other replicates
def average_averages(df1, df2, df3):
    pd.set_option('display.max_columns', None)
    df1.rename(columns={'Parameters': 'Parameters_all'}, inplace=True)
    df1.rename(columns={'Pc_change': 'Pc_change1'}, inplace=True)
    df1.rename(columns={'Error_pc_change': 'Error_pc_change1'}, inplace=True)
    df2.rename(columns={'Pc_change': 'Pc_change2'}, inplace=True)
    df2.rename(columns={'Error_pc_change': 'Error_pc_change2'}, inplace=True)
    df3.rename(columns={'Pc_change': 'Pc_change3'}, inplace=True)
    df3.rename(columns={'Error_pc_change': 'Error_pc_change3'}, inplace=True)
    master_df = pd.concat([df1, df2, df3], axis=1)
    print(master_df)
    filter_col = [col for col in master_df if col.startswith('Pc')]
    master_df["Total_average"] = master_df[filter_col].mean(axis=1)
    # master_df["Total_error"] = np.sqrt((master_df["Error_pc_change1"]**2 + master_df["Error_pc_change2"]**2
    #                                     + master_df["Error_pc_change3"]**2)/3)
    master_df["Total_error"] = (np.sqrt((master_df['Error_pc_change1']/master_df['Pc_change1'])**2
                                        + (master_df['Error_pc_change2']/master_df['Pc_change2'])**2
                                        + (master_df['Error_pc_change3']/master_df['Pc_change3'])**2
                                        ) * master_df["Total_average"]
                                )
    print(master_df)
    return master_df


def average_averages_sc(df1, df2, df3):
    pd.set_option('display.max_columns', None)
    df1.rename(columns={'Parameters': 'Parameters_all'}, inplace=True)
    df1.rename(columns={'Average': 'Average1'}, inplace=True)
    df1.rename(columns={'SEM': 'SEM1'}, inplace=True)
    df2.rename(columns={'Average': 'Average2'}, inplace=True)
    df2.rename(columns={'SEM': 'SEM2'}, inplace=True)
    df3.rename(columns={'Average': 'Average3'}, inplace=True)
    df3.rename(columns={'SEM': 'SEM3'}, inplace=True)
    master_df = pd.concat([df1, df2, df3], axis=1)
    # print(master_df)
    filter_col = [col for col in master_df if col.startswith('Average')]
    master_df["Total_average"] = master_df[filter_col].mean(axis=1)
    # master_df["Total_error"] = np.sqrt((master_df["STDEV1"]**2 + master_df["STDEV2"]**2
    #                                     + master_df["STDEV3"]**2)/3)
    master_df["Total_error"] = (np.sqrt((master_df['SEM1']/master_df['Average1'])**2
                                        + (master_df['SEM2']/master_df['Average2'])**2
                                        + (master_df['SEM3']/master_df['Average3'])**2
                                        ) * master_df["Total_average"]
                                )
    print(master_df)
    return master_df


def combine_pc_averages(df1, df2, df3):
    master_df = pd.concat([df1, df2, df3])
    print(master_df)
    return master_df


def baseline_with_mean(df):
    print("this is the df", df)
    base = df['Average'].iloc[0]
    df["Average"] -= base
    return df


# for stats


def get_percent_change_for_stats(d_list):
    # this functions converts all single values to a percent change
    modified_lists = []
    mean0 = np.mean(d_list[0])
    print("mean is", mean0)

    for d in d_list:
        new_d = []
        for item in d:
            new_item = (item / mean0)*100
            new_d.append(new_item)
        modified_lists.append(new_d)
    return modified_lists


def group_percent_change_for_stats(rep1, rep2, rep3):
    master_list = []
    for rep in rep1:
        parameter = rep1[rep1.index(rep)] + rep2[rep1.index(rep)] + rep3[rep1.index(rep)]
        master_list.append(parameter)
    return master_list


def average_averages_for_stats(master_list):
    averages = []
    for list in master_list:
        average = np.mean(list)
        averages.append(average)
    return averages
