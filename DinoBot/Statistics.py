import statistics
from statistics import mean
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats


def average_points(points):
    # get average of points and error
    average = mean(points)
    error = statistics.stdev(points)
    return average, error


def subtract_averages(average1, error1, average2, error2):
    # get difference between 2 averages and the associated error
    difference = average1-average2
    error = math.sqrt(error1**2 + error2**2)
    return difference, error


def average_tech_replicates(rep1, error1, rep2, error2, rep3, error3):
    # average technical replicates within same biological sample
    reps = [rep1, rep2, rep3]
    average = mean(reps)
    error = average * math.sqrt((error1/rep1)**2+(error2/rep2)**2+(error3/rep3)**2)
    return average, error


def divide_by_area(pc, error_pc):
    pc_per_cm2 = pc/np.pi
    error = (pc_per_cm2)*(error_pc/pc)
    return pc_per_cm2, error


def normalise_by_chla(pc, error_pc, nmol_chla):
    normalised = pc/nmol_chla
    error = (normalised) * (error_pc / pc)
    return normalised, error


def average_bio_replicates(rep1, error1, rep2, error2, rep3, error3):
    reps = [rep1, rep2, rep3]
    average = mean(reps)
    error = average * math.sqrt((error1 / rep1)**2 + (error2 / rep2)**2 + (error3 / rep3)**2)
    return average, error


def check_normality(dataset, n_bins):
    # must export data as dataframe
    df = pd.DataFrame(dataset, columns=['density'])
    # create two plots, one for the distribution, one Q-Q
    # plot the histogram
    p1 = plt.hist(df.values, bins=n_bins, edgecolor="k")
    plt.xlabel('Current density (uA cm$^{-2}$)', size=10)
    plt.ylabel('Count', size=10)
    plt.title('Distribution of data (+ Mn)')
    # plot the Q-Q plot
    p2 = sm.qqplot(df.values, line='q')
    plt.title('Q-Q plot')
    plt.show()
    print(df.describe())


def check_normality_list(list, p_threshold):
    k2, p = stats.normaltest(list)
    # p should be bigger than threshold for the distribution to be normal
    print('p-value:', p)
    if p < p_threshold:
        print("Null hyp rejected, distribution not normal")
    else:
        print("Null hyp accepted, distribution normal")


# Assumptions for two-sample paired t-test, two-tailed
# A. parent distributions must be normally distributed
# B. each data point in the samples is independent of the others
# C. no major outliers

# 1. perform normality test with Shapiro-Wilk test
# 2. do the ttest: pg.ttest(aripo, guanapo, paired=True
#          correction = False).transpose()


def unpaired_ttest(popa, popb):
    test = stats.ttest_rel(popa, popb)
    print(test)


def paired_ttest(popa, popb):
    test = stats.ttest_rel(popa, popb)
    print(test)
