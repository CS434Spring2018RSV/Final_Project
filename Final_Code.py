 # Program Filename:  Final_Code.py
 # Author: Justin Sherburne, Scott Russell, Jacob Volkman
 # Date: 6/05/18
 # Description: 
 # Input: 
 # Output: 
 
import sys
import csv
import time
import random
import numpy as np
from numpy.linalg import eig
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#from scipy import misc
#mpl.get_backend()
#import cv2
#import pylab

List1 = 'list_1.csv'
Subject1 = 'Subject_1.csv'
List2 = 'list2_part1.csv'
Subject2 = 'Subject_2_part1.csv'
List4 = 'list_4.csv'
Subject4 = 'Subject_4.csv'
List6 = 'list_6.csv'
Subject6 = 'Subject_6.csv'
List7 = 'list_7_part1.csv'
Subject7 = 'Subject_7_part1.csv'
List9 = 'list_9.csv'
Subject9 = 'Subject_9.csv'

class Algorithm:												#Algorithm class used for calculations
	def __init__(self,trainM,testM,featureNum,lammda):
		self.train_M = trainM
		self.test_M = testM
		self.addFeature(featureNum)						#Training and testing functions
		self._lammda = lammda
		

class Prog:													#Main function	
	def __init__(self):
		file_object = open(DATA) as csvfile
		lines = [line.split() for line in file_object]
		narray = np.asarray(lines)

		"""
		with open('file.csv', 'rb') as csvfile:
			filecontents = csv.reader(csvfile, delimiter=' ', quotechar='|')
			for row in filecontents:
				print ', '.join(row)
		"""
		
		y=0
		v=0
		for k in lines:
			for i in k:
				my_list = i.split(",")
				for p in my_list:
					self.Matrix[y][v] = int(p)
					# print(y)
					# print("' ")
					# print(v)
					v += 1
				y += 1
				v = 0
		#Matrix is addressable by Matrix[6000][784] 
		#print(Matrix)
				

		

#-----------------------
#		Main
#-----------------------

if __name__ == "__main__":
	prog = Prog()
	argc = len(sys.argv)
	prog.question_1()	
