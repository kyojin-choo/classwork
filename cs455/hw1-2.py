### hw_455.py -*- CMSC455: Numerical Computations Homework -*-
## 
### Author: Daniel Choo
### Date:   10/20/19

import os
import sys
import time
import math
import numpy as np
from numpy import array
from numpy import polyfit
from numpy import polyval

def takeoff():
	""" takeoff(): hw1; prints every 0.1 seconds; computes maximum height of a rocket 
	               when fired straight up. It will print: time (s), height (m), v (m/s), a (m/s^2)
	               m (kg), and will stop when it reaches max height.

      Returns:   N/A
	"""
	# Initial conditions. 
	check = False
	counter = 0
	t = 0.0                # time (t)
	dt = 0.1               # delta time (t)
	s = 0.0                # position (m)
	v = 0.0                # velocity (m/s)
	dv = 0.0               # velocity change post-launch off 
	a = 0.0                # acceleration (m/s^2)
	F = 0.0                # total force, not including gravity. (N)
	g = 9.80665            # gravity (m/s^2)
	Rho = 1.293            # density of air (kg/m^3)
	Ft =  [0.0, 9.0, 14.09, 8.0, 5.5, 5.01, 4.2, 4.8, 4.0, 5.9, 3.6, 3.4, 4.1, 2.5, 3.28, 3.72, 3.9, 3.7, 1.3, 0.0] 
	m_height = 0           # max height (m)
	m = 0.0340 + 0.0242    # total mass (g)

  # Rocket
	body_A = 0.000506      # body surface area (sq m)
	fin_A = 0.00496        # fin surface area (sq m) 
	fin_Cd = 0.01          # coefficient of drag (dimensionless) (fin cd)
	body_Cd = 0.45         # coefficient of drag (dimensionless) (body cd)
	engine_m = 0.0242      # mass of engine
	f_eng_m = 0.0094       # final mass of engine
	body_m = 0.0340        # mass of body

	while v >= 0:
		fin_Fd = fin_Cd * Rho * fin_A * v**2 / 2          # Force of drag (fin)
		body_Fd = body_Cd * Rho * body_A * v**2 / 2       # Force of drag (body)

		if t == 0:                                        # Stationary @ t=0
			m += 0
		elif m > (body_m + f_eng_m):                      # If mass is > than no fuel engine+body
			m -= 0.0001644*Ft[counter]
		else:                                             # Cannot be smaller than no fuel engine.
			m = body_m + f_eng_m

		Fg = m*g                                          # Force of gravity.
		F = Ft[counter] - (fin_Fd + body_Fd + Fg)         # Total force.
		a = F/m                                           # Acceleration
		dv = a*dt                                         # Change in velocity.
		v+=dv                                             # New velocity

		# If executed at t = 0, should be stationary.
		if t == 0:
			v = 0

		ds = v*dt                                         # Change in distance.
		s+=ds                                             # New distance.

		# print!
		print("\nCurrent time: " + str(t))
		print("Current height (m): " + str(s))
		print("Current velocity (m/s): " + str(v))
		print("Current acceleration (m/s^2): " + str(a))
		print("Current mass (kg): " + str(m))

		t+=dt                                              # Increment time 0.1s
		
		if counter < len(Ft)-1:                            # If counter < 19, increment.      
			counter+=1
		else:                                              # If == 19, then stay at 19.
			counter = len(Ft)-1
		#end of loop

def lsfit():
	""" lsfit(): hw2: will fit the least square polynomial.
               x (independent) = time
	             y (dependent) = thrust
	             We will be calculating max error, avg error, and rms error.
	
	    Returns: N/A
	"""
	print("\nComputing Least Square Fit..." )
	
	time = array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])       # time (independent)
	thrust =  array([0.0, 9.0, 14.09, 8.0, 5.5, 5.01, 4.2, 4.8, 4.0, 5.9, 3.6, 3.4, 4.1, 2.5, 3.28, 3.72, 3.9, 3.7, 1.3, 0.0]) # thrust (dependent)

	# "Do the least square fit with 3, 4, ... , 17 degree polynomials."
	for i in range(3, 18):
		print("\nCurrent polynomial: " + str(i) + "\n------------")

		# Utilizing numpy's polyfit and polyval rather than the one provided in the lecture notes.
		pf = polyfit(time, thrust, i)            # Returns polynomical coefficients 
		pv = polyval(pf, time)                   # Returns y = p(x) for n values

		err = abs(thrust-pv)                    
		err_summation = sum(err)                 # Truncating the loop found in least_square.py3; avg_error summed the error array and divided by the length of the array. 
		err_sum_sq = 0.0

		# Sum of value @ i squared
		for i in range(err.size):
			temp = err[i]
			err_sum_sq+=temp*temp

		max_err = max(abs(pv - thrust))          # Returns max value in array subtraction 
		print("Max Error: " + str(max_err))    

		avg_err = err_summation/(len(time))      # Sum divided by the array len to get avg.
		print("Avg Error: " + str(avg_err))

		rms_err = math.sqrt(err_sum_sq/len(time)) # Sum of value (^2ed) divided by len of array
		print("RMS Error: " + str(rms_err))
		
def menu(usr_input):
	""" menu():  just prints a menu.
	    Returns: usr_input (int)
  """

	while usr_input <= 0  or usr_input > 8:
		usr_input = int(input("\nChoose an option: \n1. Calculate maximum height.\n2. Least Square Fit\n8. Exit\n\n"))

	return usr_input

def main():
	""" main():  bootstrapper.
	    Returns: N/A
	"""
	usr_input = 0

	print("\nWelcome to the Rocket Simulator.\n----------------")


	while usr_input != 8:
		usr_input = menu(0)
		
		if usr_input == 1:
			takeoff()
		
		elif usr_input == 2:
			lsfit()

		elif usr_input == 3:
			pass

		elif usr_input == 8:
			sys.exit()
			
		else:
			pass

main()
