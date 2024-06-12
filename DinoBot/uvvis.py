import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_file(file):
    with open(file) as t:
        contents_unfiltered = t.readlines()
        contents = []
        for row in contents_unfiltered:
            contents.append(row)
        print(contents[0])
        contents.pop(0)
        print(contents[0])
        contents.pop(0)
        wavelength = []
        absorbance = []
        for line in contents:
            wavelength.append(float(line.split(',')[0]))
            absorbance.append(float(line.split(',')[1]))
        return wavelength, absorbance


def create_df(wavelength, absorbance):
    dict = {'Wavelength (nm)': wavelength, 'Absorbance': absorbance}
    df = pd.DataFrame(dict)
    return df


def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


def linechart(df):

    clim = (400, 800)
    norm = plt.Normalize(*clim)
    wl = np.arange(clim[0], clim[1] + 1, 2)
    colorlist = list(zip(norm(wl), [wavelength_to_rgb(w) for w in wl]))
    spectralmap = mpl.colors.LinearSegmentedColormap.from_list("spectrum", colorlist)

    fig, axs = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)

    # plots the line
    wavelengths = df["Wavelength (nm)"]
    spectrum = df["Absorbance"]
    plt.plot(wavelengths, spectrum, color='black')

    # makes a matrix out of the y space from 0 to 1 and all of X
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(wavelengths, y)

    extent = (np.min(wavelengths), np.max(wavelengths), np.min(y), np.max(y))

    plt.imshow(X, clim=clim, extent=extent, cmap=spectralmap, aspect='auto')
    plt.xlabel('Wavelength (nm)', size=12)
    plt.ylabel('Absorbance', size=12)
    plt.title('Absorption spectrum of $psbI$ mutant')

    plt.fill_between(wavelengths, spectrum, 1, color='w')
    # plt.savefig('WavelengthColors.png', dpi=200)

    plt.show()


""" Example run """

# read file
wl, ab = read_file("/Users/lorismarcel/PycharmProjects/uvvis/2024-02-21 UV-vis psbE:psbI copy/psbI-halfdiluted.csv")

# create a data frame
df = create_df(wl, ab)
print(df)

# sort and plot
df2 = df.sort_values(['Wavelength (nm)'], ascending=True)
linechart(df2)



