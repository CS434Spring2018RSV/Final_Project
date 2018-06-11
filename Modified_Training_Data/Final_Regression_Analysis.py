#cs 434 Scott Russell and Vinayaka Thompson
# Assignment 1: Linear Regression:

#numpy is used for matpltlib
import numpy as np

#basic import matplot to graph AES distribution
import matplotlib
matplotlib.use('Agg')

#define a variable to acess graph functionality (plt)
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt




def main():

	#call training for weight vector (with and without Dummy)
    dummy = 1;
    noDummy = 0;
    # Linear regression with dummy variable
    #print "\n WITH DUMMY VARIBLE \n"
    #Calculation(dummy)
    # Linear regression without dummy variable
    print "\n Testing for Final Report\n"
    Calculation(noDummy)

   # print "\n RANDOM DATA DISTRIBUTION (0-20 even only) \n"
    #normDataTrain()

def Calculation(dummy):


	#training Dataset
	#File Selection (under data subfolder)
    fileName = loadD("./data/ReadTrain1.csv")


	#if Dummy Variable is actiaveted or not
    if dummy:
        X = np.insert(fileName[:, 0:-1], 0, 1, axis=1)
    else:
        X = fileName[:, 0:-1]

	#For both identify weight vector
	Y = fileName[:, -1]

	w = calcWeight(X, Y)

	print "Y is:"
	print Y
	print "Weight Vector:"
	print w
	
	calcASE(w, dummy)

def normDataTrain():
    trainASE = []
    testASE = []
    randomSize = 22

    for d in range(0, randomSize, 2):
        fileName = loadD("./data/ReadTrain1.csv")
        X = np.insert(fileName[:, 0:-1], 0, 1, axis=1)
        Y = fileName[:, -1]
        n = X.shape[0]

        fileName = loadD("./data/ReadTrain2.csv")
        testX = np.insert(fileName[:, 0:-1], 0, 1, axis=1)
        testY = fileName[:, -1]
        testN = testX.shape[0]

        # add random data from normal distribution
        for i in range(d):
            mu = np.random.rand() * 1000 + 1000 #selection of random seed)
            sigma = np.random.rand() * 1000 + 1000
            normalData = np.array(np.random.normal(mu, sigma, n))
			#normal data distribution based on seeds and size of random variables
            X = np.insert(X, X.shape[1], normalData, axis=1)
            normalTestingData = np.random.normal(mu, sigma, testN)
            testX = np.insert(testX, testX.shape[1], normalTestingData, axis=1)
			#tests based on data 
		#weight calculation based on x and y 
        w = calcWeight(X, Y)
		#normal AES calculation (using normalization of SSE with length of variable)
        trainASE.append(calcASEwithNorm(X, Y, w))
        testASE.append(calcASEwithNorm(testX, testY, w))

    print "Training ASE: (11 Values)"
    print trainASE
    print "Testing ASE: (11 values)"
    print testASE

	#Graph both training and testing on same Axis with size ==randomsize (22)
    graphAESplt(trainASE, testASE, range(0, randomSize, 2))

def calcWeight(X, Y):
    # Calculates the weight vector
    Xtrans = np.transpose(X)
    a = np.linalg.inv(np.matmul(Xtrans, X))
    b = np.matmul(Xtrans, Y)
    w = np.matmul(a, b)
	#return weight calculation (matmul used for calculations)
    return w

	
def calcASE(w, dummyVar):
    # calculate ASE for training & test data
    files = ["./data/ReadTrain1.csv", "./data/ReadTrain2.csv"]
    names = ["Training data ASE: ", "Testing data ASE: "]
    w = np.transpose(w)

    for idx, f in enumerate(files):
        fileName = loadD(f)

        if dummyVar:
            X = np.insert(fileName[:, 0:-1], 0, 1, axis=1)
        else:
            X = fileName[:, 0:-1]
        Y = fileName[:, -1]

        result = np.dot(X, w)
        sqDiffs = (result - Y) ** 2
        SSE = sum(sqDiffs)
        ASE = SSE / len(sqDiffs)

        print names[idx]
        print ASE

#Calculate AES with normalization of SSE/length
def calcASEwithNorm(X, Y, weight):
    result = np.dot(X, weight)
    squaredDiffs = (result - Y) ** 2	
	#First calulate SSE then use Normalization to calculated with SSE dividied by the length
    SSECalc = sum(squaredDiffs)
    ASECalc = SSECalc / len(squaredDiffs)
    return ASECalc

	
	#use poplot (plt) to graph AES Data
def graphAESplt(trainASE, testASE, dRange):
    # plot training error
	plt.plot(dRange, trainASE, label="train")
	plt.xlabel('Number of Random Features')
	plt.ylabel('ASE Calculation')
	#plot testing error
	plt.plot(dRange, testASE, label="test")
	plt.xlabel('Number of Random Features')
	plt.ylabel('ASE Calculation')
	plt.savefig('GraphAESData.png')


def loadD(fileName):
	#loading data
    file = open(fileName, "r")
    data = np.genfromtxt(file, delimiter=",")
	#Return the matrix with delimeter spaces
    return data


#for errors with module access
if __name__ == "__main__":
	main()
