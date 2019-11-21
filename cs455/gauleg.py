# gauleg.py Gauss Legendre numerical quadrature, x and w computation 
# integrate from a to b using n evaluations of the function f(x)  
# usage: from gauleg import gaulegf         
#        x,w = gaulegf( a, b, n)                                
#        area = 0.0                                            
#        for i in range(1,n+1):          #  yes, 1..n                   
#          area += w[i]*f(x[i])                                    

import math
import numpy as np

def gaulegf(a, b, n):
  x = np.array([0.0 for i in range(n+1)]) # x[0] unused
  w = np.array([0.0 for i in range(n+1)]) # w[0] unused
  eps = 3.0E-14
  m = int(float((n+1)/2))
  m1 = m+1
  xm = 0.5*(b+a)
  xl = 0.5*(b-a)
  for i in range(1,m1):
    z = math.cos(3.141592654*(i-0.25)/(n+0.5))
    while True:
      p1 = 1.0
      p2 = 0.0
      for j in range(1,n+1):
        p3 = p2
        p2 = p1
        p1 = ((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j

      pp = n*(z*p1-p2)/(z*z-1.0)
      z1 = z
      z = z1 - p1/pp
      if abs(z-z1) <= eps:
        break

    x[i] = xm - xl*z
    x[n+1-i] = xm + xl*z
    w[i] = 2.0*xl/((1.0-z*z)*pp*pp)
    w[n+1-i] = w[i]
  return x, w
# end gaulegf 

