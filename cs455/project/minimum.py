# minimum.py -*- finding the global minimum of a function -*-
#
# Author: Daniel Choo
# Date:   12/06/19

import numpy as np
import math
from mpmath import mp, mpf, sin, exp


def calc(x, y):
    """ calc(x, y): takes in x, y and calculates the z function
                    (multiple precision, 100 digits).
        Returns:    z function (float)
    """
    mp.dps = 100
    mpf.dps = 100
    return exp(sin(50.0*x)) + sin(60.0*exp(y)) + sin(70.0*sin(x)) + sin(sin(80.0*y)) - sin(20.0*(x+y)) + (x*x+y*y)/4.0


def optimize(x_best, y_best, z_best):
    """ optimize(x, y, z): Will attempt to find the global minimum of the z func.
        Returns:           z (multiple precision float)
    """
    print("\nFinding global minimum...")
    step = 0.0001
    iterate = 1E-100

    while step >= iterate:
        # Checking up and down on y.
        temp_x = x_best
        temp_y = y_best - step
        z = calc(temp_x, temp_y)
        if z < z_best:
            y_best = temp_y
            z_best = z

        temp_x = x_best
        temp_y = y_best + step
        z = calc(temp_x, temp_y)
        if z < z_best:
            y_best = temp_y
            z_best = z

        # Checking side to side
        temp_x = x_best - step
        temp_y = y_best
        z = calc(temp_x, temp_y)
        if z < z_best:
            x_best = temp_x
            z_best = z

        temp_x = x_best + step
        temp_y = y_best
        z = calc(temp_x, temp_y)
        if z < z_best:
            x_best = temp_x
            z_best = z

        # Checking four corners.
        temp_x = x_best + step      # Top right corner
        temp_y = y_best + step
        z = calc(temp_x, temp_y)
        if z < z_best:
            x_best = temp_x
            y_best = temp_y
            z_best = z

        temp_x = x_best - step      # Top left corner
        temp_y = y_best + step
        z = calc(temp_x, temp_y)
        if z < z_best:
            x_best = temp_x
            y_best = temp_y
            z_best = z

        temp_x = x_best - step      # Bottom left corner
        temp_y = y_best - step
        z = calc(temp_x, temp_y)
        if z < z_best:
            x_best = temp_x
            y_best = temp_y
            z_best = z

        temp_x = x_best + step      # Bottom right corner
        temp_y = y_best - step
        z = calc(temp_x, temp_y)
        if z < z_best:
            x_best = temp_x
            y_best = temp_y
            z_best = z

        step /= 2

    return x_best, y_best, z_best


def main():
    """ main():  bootstrapper to calculate global min of a z function.
        Returns: 0 (int): successful exit
    """
    print("Beginning to find the global minimum for the function:")
    print("exp(sin(50.0*x)) + sin(60.0*exp(y)) + sin(70.0*sin(x)) + sin(sin(80.0*y)) - sin(20.0*(x+y)) + (x*x+y*y)/4.0")
    step = 0.001
    x_best = 0.0
    y_best = 0.0
    z_best = 0.0
    interval = np.arange(-1, 1, step)

    for x in interval:
        for y in interval:
            z = math.exp(math.sin(50.0*x)) + math.sin(60.0*math.exp(y)) + math.sin(70.0*math.sin(x)) + math.sin(math.sin(80.0*y)) - math.sin(20.0*(x+y)) + (x*x+y*y)/4.0
            if z < z_best:
                z_best = z
                x_best = x
                y_best = y

    print("\nFound local minimum from (-1, 1): " + str(z_best))
    x_best, y_best, z_best = optimize(x_best, y_best, z_best)

    # Time to calculate multiple precision.
    z_best = calc(x_best, y_best)

    print("\nx = " + str(x_best) + ", y = " + str(y_best) + ", z = " + str(z_best))
    return 0


main()
