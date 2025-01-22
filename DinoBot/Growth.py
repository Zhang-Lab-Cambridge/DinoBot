import pandas as pd


def read_data(file):
    with open(file) as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row.rstrip())
        contents.pop(0)
        day = []
        av_count = []
        stdv_count = []
        av_chl = []
        stdv_chl = []
        av_OD = []
        stdv_OD = []
        for line in contents:
            day.append(float(line.split('\t')[0]))
            av_count.append(float(line.split('\t')[1]))
            stdv_count.append(float(line.split('\t')[2]))
            av_chl.append(float(line.split('\t')[3]))
            stdv_chl.append(float(line.split('\t')[4]))
            av_OD.append(float(line.split('\t')[5]))
            stdv_OD.append(float(line.split('\t')[6]))
        return day, av_count, stdv_count, av_chl, stdv_chl, av_OD, stdv_OD


def read_calibration_data(file):
    with open(file) as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row.rstrip())
        contents.pop(0)
        count = []
        chl = []
        OD = []
        for line in contents:
            count.append(float(line.split('\t')[0]))
            chl.append(float(line.split('\t')[1]))
            OD.append(float(line.split('\t')[2]))
        return count, chl, OD


def create_growth_df(day, av_count, stdv_count, av_chl, stdv_chl, av_OD, stdv_OD):
    dict = {'Day': day, 'Av_count': av_count, 'Stdv_count': stdv_count, 'Av_chl': av_chl, 'Stdv_chl': stdv_chl,
            'Av_OD': av_OD, 'Stdv_OD': stdv_OD}
    df = pd.DataFrame(dict)
    return df


def create_calibration_df(count, chl, OD):
    dict = {'Count': count, 'Chl': chl, 'OD': OD}
    df = pd.DataFrame(dict)
    return df


def extract_growth_data(file):
    a, b, c, d, e, f, g = read_data(file)
    df = create_growth_df(a, b, c, d, e, f, g)
    return df


def extract_calibration_data(file):
    a, b, c = read_calibration_data(file)
    df = create_calibration_df(a, b, c)
    return df

