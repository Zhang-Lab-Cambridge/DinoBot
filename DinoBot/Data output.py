from Photocurrents import *
from Graphing import *
from Light_modulation import *
from Stepped_chronoamperometry import *
from Statistics import *
from CV_analysis import *


""" Use this file to run code, keep the other files clean """


""" How to analyse photocurrents """

# analyse the data with this function:
# data, list_of_changes, densities = analyse_photocurrents("filepath", light_first_on, nmol_chla, time_on, time_off)

# plot the data with light-dark rectangles
# ld_line_chart(data, list_of_changes, Title)



""" How to analyse stepped chronoamperometry """


# analyse the data with this function:
# data, list_of_changes, densities = analyse_stepped_chrono("filepath", light_first_on, nmol_chla, time_on, time_off)

# plot the data with light-dark rectangles
# ld_line_chart(data, list_of_changes, Title)

# use box plots to compare current densities at different potentials
# sc_replicate_to_3(densities)
# list_of_potentials = []
# comparative_boxplot(densities, list_of_potentials)


