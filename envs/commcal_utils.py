from math import log10


def mW2dBm(mW):
    return 10. * log10(mW)


# Function to convert from dBm to mW
def dBm2mW(dBm):
    return 10 ** ((dBm) / 10.)


def convert_mhz_to_m(wavelength_mhz):
    speed_of_light = 299792458  # meters per second
    wavelength_m = speed_of_light / (wavelength_mhz * 10 ** 6)
    return wavelength_m


if __name__ == '__main__':
    speed_of_light = 299792458  # meters per second
    frequency_mhz = 2500  # frequency in megahertz

    wavelength_m = speed_of_light / (frequency_mhz * 10 ** 6)
    print(wavelength_m)
