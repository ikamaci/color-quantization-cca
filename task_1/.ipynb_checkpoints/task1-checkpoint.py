from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

def plotValues(data, xAxis=[], xLabel="", yLabel="",title=""):
    plt.xlabel(xLabel)
    plt.ylabel(yLabel) 
    plt.title(title) 
    if len(xAxis) > 0:
        plt.plot(xAxis, data, '-ok')
    else:
        plt.plot(data, '-ok')

def clusterize(centroids, data, k):
    clusters = [ [] for i in range(k)]
    labels = []
    for dataPoint in data:
        cluster_index=-1
        distance = float('inf')
        for c_index, cluster_center in enumerate(centroids):
            #print("cluster center distance", len(cluster_center))
            distance_to_center = calculateDistance(cluster_center, dataPoint)
            if distance_to_center < distance:
                distance = distance_to_center
                cluster_index = c_index
        labels.append(cluster_index)
        #clusters[cluster_index].append(np.array(dataPoint))
    return labels  


def calculateDistance(pointA, pointB):
    #if len(pointA) != len(pointB):
        #print(len(pointA),len(pointB))
    substract= np.subtract(pointA, pointB)
    return np.linalg.norm(substract)

def calculateCentroids(labels,data,k):
    #print(len(labels),len(data))
    clusters = [[] for i in range(k)]
    #print("clusters",len(clusters))
    for index,point in enumerate(data):
        clusterIndex = int(labels[index])
        clusters[clusterIndex].append(np.array(point))
    #print(np.shape(clusters[0]), np.shape(clusters[1]))
    newCentroids = []
    for cluster in clusters:
        clusterCenter = np.mean(cluster,axis=0)
        #print("cluster",len(cluster), np.shape(np.asarray(clusterCenter)))
        newCentroids.append(clusterCenter)
    return newCentroids, clusters


def kMeans(kValue, data, initial_centroids):
    lossValues = []
    labels = clusterize(initial_centroids, data, kValue)
    newCentroids = initial_centroids
    stop= False
    previousLoss = float('inf')
    iteration = 0
    thresold = 0.001
    while not stop:
        iteration += 1
        print(iteration)
        newCentroids, clusters = calculateCentroids(labels,data,kValue)
        #print("newCentroids")
        #print(newCentroids)
        labels = clusterize(newCentroids, data, kValue)
        currentLoss = 0
        for c in clusters:
            clusterVar = clusterVariance(c)
            currentLoss += clusterVar
        if(((previousLoss - currentLoss)/currentLoss) < thresold):
            stop = True
        lossValues.append(currentLoss)
        previousLoss = currentLoss
    return labels, newCentroids, lossValues

def clusterVariance(data):
    center = np.mean(data,axis=0)
    variance = 0
    for dataPoint in data:
        distanceSquare = np.square(calculateDistance(dataPoint, center))
        variance += distanceSquare
    return variance

def generateRandomCenters(k=2):
    centers = []
    for i in range(k):
        centers.append([np.random.uniform(250) for i in range(3)])
    return centers

def constructImageMatrix(labels, centroids, shape):
    resultNormalized = []
    for label in labels:
        resultNormalized.append(np.asarray(centroids[label]))
    return np.asarray(resultNormalized).reshape(shape)

def generateImageFromMatrix(imageMatrix, file_name = 'my_result.jpeg'):
    imgResult = Image.fromarray(imageMatrix.astype(np.uint8), 'RGB')
    imgResult.save(file_name)

def kMeansRun(dataset, init, k,inputFile, outfileName):
    dataNormalized = dataset.reshape((len(dataset)*len(dataset[0])),3)
    if init == 'random':
        initial_centroids = generateRandomCenters(k)
    else:
        im = Image.open(inputFile)
        plt.imshow(im)
        points = plt.ginput(k, show_clicks=True) 
        initial_centroids = np.asarray(points)
    initial_centroids = generateRandomCenters(k)
    resultLabels, resultCentroids, lossValues = kMeans(k,dataNormalized, initial_centroids)
    imageMatrix = constructImageMatrix(resultLabels, resultCentroids, np.shape(dataset))
    generateImageFromMatrix(imageMatrix, file_name=inputFile +" k = "+str(k)+" quantized.jpeg")
    return resultLabels, lossValues


def __main__():
    k = int(sys.argv[1])
    fileName = sys.argv[2]
    init = 'random'
    if len(sys.argv) >= 3:
        init = sys.argv[3]
    image = Image.open(fileName)
    dataset = np.asarray(image)
    resultingLabels, lossData = kMeansRun(dataset, init, k, inputFile=fileName,  outfileName="4quantized")

__main__()