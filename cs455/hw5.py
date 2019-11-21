# hw5.py -*- craziness -*-
#
# Author: Daniel Choo
# Date:   11/17/19

import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft, ifft


def main():
    """  main(): speeds up the .wav file.
         returns: 0 [int]: success
    """
    usr_input = ""

    while True:
        usr_input = input("Which .wav file would you like to pitch shift?: ")
        usr_input += ".wav"

        # Attempt to open the file. If not possible, wrong file or DNE.
        try:
            print("\nAttempting to read in " + usr_input)
            rate, data = wavfile.read(usr_input, "r")     # Read in .wav
            break

        # Fails to break.
        except FileNotFoundError:
            print("File does not exist.\n")

    print("Successfully read in " + usr_input + "\n")

    left = data[0::2]             # Always mono
    shift = 5                     # Arbitrary number
    fft_left = fft(left)          # FFT
    np.roll(fft_left, shift)      # Will roll the array to the right 5.
    fft_left[0:shift] = 0         # Set the first five to 0
    ifft_left = ifft(fft_left)    # IFFT

    # Write out.
    wavfile.write("output.wav", rate, ifft_left.astype(data.dtype))
    return 0


main()
