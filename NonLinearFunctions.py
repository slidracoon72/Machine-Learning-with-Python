# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 13:12:03 2020

@author: Rahul
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)


#LINEAR function
##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
#plt.figure(figsize=(8,6))
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.title("Linear")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

#NON-LINEAR function
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.title("Cubic")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

#Quadratic
x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

y = np.power(x,2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.title("Quadratic")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

#Exponential
X = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph

Y= np.exp(X)

plt.plot(X,Y) 
plt.title("Exponential")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

#Logarithmic
X = np.arange(-5.0, 5.0, 0.1)

Y = np.log(X)

plt.plot(X,Y) 
plt.title("Logarithmic")
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

#Sigmoidal/Logistic
X = np.arange(-5.0, 5.0, 0.1)


Y = 1-4/(1+np.power(3, X-2))

plt.plot(X,Y)
plt.title("Sigmoidal/Logistic") 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()