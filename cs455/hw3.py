### hw3.py -*- integration -*-
##
### Author: Daniel Choo
### Date:   08/13/19

import sys
import math
import numpy as np
from gauleg import gaulegf as gl
 
def c1(x,y):
    """ c1(x,y): circle 1
        Return:  pos on curve
    """
	return ((x-2)**2 + (y-2)**2)

def c2(x,y):
    """ c2(x,y): circle 2
        Return:  pos on curve
    """
	return x**2 + (y-2)**2 

def c3(x,y):
	""" c3(x,y): circle 3
        Return: pos on curve
    """
	return x**2 + y**2

def q2():
	""" q2():   Counting the dots in the area and multiplying the count by the 
	            grid squared. This provides the Area between C2 and C3, while 
	            removing C1's area.

	    Return: N/A
	"""
	
	grid = [0.1, 0.01, 0.001]

	print("\nQuestion 2: Area Between Three Circles\n")
	
	for i in range(0,3):
		x = -3
		points = 0

		while x <= 3:
			y = -4

			while y <= 4:
				if (c1(x,y) >= 1.0**2.0) and (c2(x,y) <= 2.0**2.0) and (c3(x,y) <= 3.0**2.0):
					points+=1
				y+=grid[i]
				# End of y
			x+=grid[i]
			# End of x

		area = points * (grid[i]**2)
		print("For a grid (" + str(grid[i]) + "), the area is " + str(area))
		# End of i

def q1b():
	""" q1b():  using gauss legendre's method to calculate the error of
	            sine integration from 0 to 1. Using points 8 and 16.

	    Return: N/A
	"""
	print("\nQuestion 1B: Gauss Legendre")
	for i in range(1,3):
		p = 8*i
		x, w = gl(0.0, 1.0, p)
		exact = 1.0 - math.cos(1.0)
		area = 0.0

		print("\nPoint = " + str(p))

		for j in range(1, p+1):
			area+=w[j] * math.sin(x[j])
			
		print("Area (Exact): " + str(exact))
		print("Area (Estimate): " + str(area))
		print("Error (Exact-Est): " + str(math.fabs(area-exact)))

def q1a():
	""" q1b():  using trapezoidal method to calculate the error of
	            sine integration from 0 to 1. Using points 16, 32, 64, and 128

	    Return: N/A
	"""
	print("\nQuestion 1A: Trapazoidal Error")
	i = 16

	while i <= 128:
		exact = 1.0 - math.cos(1.0)
		area = (math.sin(0.0) + math.sin(1.0)) / 2.0
		h = 1/i
		
		for j in range(1, i-1):
			area += math.sin(j*h)

		area*=h
		error = area - exact

		print("\nPoint = " + str(i))
		print("Area (Exact) = " + str(exact))
		print("Area (Estimate) = " + str(area))
		print("Error (Est-Exact): " + str(math.fabs(error)))
		i*=2

def menu(x):
	"""menu():  menu
       Return:  x (int: user input)
	"""
	while x <= 0 or x > 4:
		x = int(input("\nPlease select an option:\n1.) Q.1A\n2.) Q.1B\n3.) Q.2\n4.) Exit\n\nInput: "))
	return x

def main():
	""" main(): bootstrapper

	    Return: N/A. Exits on condition
	"""
	usr_input = 0

	while usr_input != 4:
		usr_input = menu(0)

		if usr_input == 1:
			q1a()
		elif usr_input == 2:
			q1b()
		elif usr_input == 3:
			q2()
		elif usr_input == 4:
			sys.exit(0)

main()
