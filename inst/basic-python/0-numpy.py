#!/bin/python2

# Load data from csv
from numpy import genfromtxt
df = genfromtxt('mtcars_ts.csv', delimiter=';')

# See dimensions of data frame
df.shape

