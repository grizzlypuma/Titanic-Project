print("solve HackerRank problems")

"""
Question:

You are given four points A, B, C and D in a 3-dimensional Cartesian coordinate
system. You are required to print the angle between the plane made by the
points A, B, C and B, C, D in degrees(not radians). Let the angle be PHI.

Cos(PHI) = (X.Y)/|X||Y| where X = AB x BC and Y = BC x CD.

Here, X, Y means the dot product of X and Y, and AB x BC means the cross
product of vectors AB and BC. Also, AB = B - A.

Input Format:
    One line of input containing the space separated floating number values of
    the X, Y and Z coordinates of a point.

Output Format:
    Output the angle correct up to two decimal places.

Sample Input:
    0 4 5
    1 7 6
    0 5 9
    1 7 2

Sample Output:
    8.19
"""

"****************************SOLUTION**************************"

import math

class Points(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __sub__(self, other):
        return Points(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        '''
            This methods does blah blah blah

            Inputs:
                other (str): description of the other parameter

            Returns:
                Points: 3 points ... 
        '''
        return Points(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def absolute(self):
        return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)

# This tool computes the cartesian product of input iterables.
# It is equivalent to nested for-loops.
# For example, product(A, B) returns the same as ((x,y) for x in A for y in B).

# Task
# You are given a two lists A and B. Your task is to compute their cartesian product X.

# Example
# A = [1, 2]
# B = [3, 4]
# AxB = [(1, 3), (1, 4), (2, 3), (2, 4)]
# Note: A and B are sorted lists, and the cartesian product's tuples should be output in sorted order.

# Input Format
# The first line contains the space separated elements of list A.
# The second line contains the space separated elements of list B.

# Both lists have no duplicate integer elements.

# Constraints
# 0 < A < 30
# 0 < B < 30

# Output Format
# Output the space separated tuples of the cartesian product.

# Enter your code here. Read input from STDIN. Print output to STDOUT

from itertools import product

a = list(map(int, input().split(" ")))
# TODO: remove this whitespace
b = list(map(int, input().split(" ")))

x = list(product(a, b))

for i in x:
    print(i, end=" ")
