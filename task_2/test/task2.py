from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

def calculateCluster(dataPoint):
    black = np.asarray([0,0,0])
    white = np.asarray([255,255,255])
    substractBlack= np.subtract(dataPoint, black)
    differenceToBlack = np.linalg.norm(substractBlack)
    substractWhite= np.subtract(dataPoint, white)
    differenceToWhite = np.linalg.norm(substractWhite)
    if differenceToBlack < differenceToWhite:
        return black
    else:
        return white


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return np.asarray(gray)

def generateImageFromMatrix(imageMatrix, file_name = 'my_result1.png'):
    imgResult = Image.fromarray(imageMatrix.astype(np.uint8))
    imgResult.save(file_name)

def toGray(data):
    grayIm = np.zeros(np.shape(data))
    for row in range(len(data)):
        for column in range(len(data[row])):
            greyIm[row][column] = rgb2gray(dataset[row][column][:])
    return grayIm

def rgb2Binary(data):
    binaryData = np.zeros(np.shape(dataset))
    for row in range(len(data)):
        for column in range(len(data[row])):
            binaryData[row][column] = calculateCluster(dataset[row][column][:])
    return binaryData

def gray2binary(data,threshold):
    binaryData = np.zeros(np.shape(data))
    for row in range(len(data)):
        for column in range(len(data[row])):
            binaryData[row][column] = 255 if data[row][column] > threshold else 0
    return binaryData

def inverse(data):
    binaryData = np.zeros(np.shape(data))
    for row in range(len(data)):
        for column in range(len(data[row])):
            binaryData[row][column] = 255 if data[row][column] == 0 else 0
    return binaryData

def gray2truncated(data,threshold):
    binaryData = np.zeros(np.shape(dataset))
    for row in range(len(data)):
        for column in range(len(data[row])):
            binaryData[row][column] = 255 if data[row][column] > threshold else data[row][column]
    return binaryData

def connected_neigbours(dataBin):
    enhanced = np.zeros((len(dataBin)+ 2, len(dataBin[0]) +2))
    firstPass = np.zeros((len(dataBin)+ 2, len(dataBin[0]) +2))
    enhanced[1:np.shape(dataBin)[0] +1,1:np.shape(dataBin)[1] +1] = dataBin
    counter = 1;
    black = 1
    for row in range(1,len(enhanced)):
        for column in range(1,len(enhanced[row])):
            if(enhanced[row][column] == black):
                find = False
                if(enhanced[row][column-1] == black):
                    find = True
                if(enhanced[row-1][column-1] == black):
                    find = True
                if (enhanced[row-1][column] == black):
                    find = True
                if (enhanced[row-1][column+1] == black):
                    find = True
                
def erosion(dataNormal):
    dataBin = inverse(dataNormal)
    enhanced = np.zeros((len(dataBin)+ 2, len(dataBin[0]) +2))
    result = dataBin[:]
    firstPass = np.zeros((len(dataBin)+ 2, len(dataBin[0]) +2))
    enhanced[1:np.shape(dataBin)[0] +1,1:np.shape(dataBin)[1] +1] = dataBin
    counter = 1;
    black = 0
    white = 255
    for row in range(1,len(enhanced)):
        for column in range(1,len(enhanced[row])):
            if(enhanced[row][column] == white):
                if(enhanced[row][column-1] == white and enhanced[row][column+1] == white and enhanced[row-1][column] == white and enhanced[row+1][column] == white):
                    result[row-1][column-1] = white
                else:
                    result[row-1][column-1] = black
    return inverse(result)


def dilation(dataBin):
    enhanced = np.zeros((len(dataBin)+ 2, len(dataBin[0]) +2))
    result = dataBin[:]
    firstPass = np.zeros((len(dataBin)+ 2, len(dataBin[0]) +2))
    enhanced[1:np.shape(dataBin)[0] +1,1:np.shape(dataBin)[1] +1] = dataBin
    counter = 1;
    black = 0
    white = 255
    for row in range(1,len(enhanced)):
        for column in range(1,len(enhanced[row])):
            if(enhanced[row][column] == white):
                if(enhanced[row][column-1] == white and enhanced[row][column+1] == white and enhanced[row-1][column] == white and enhanced[row+1][column] == white):
                    result[row-1][column-1] = white
                else:
                    result[row-1][column-1] = black
    return result

def __main__():
    fileName = "input.jpg"
    image = Image.open(fileName)
    dataset = np.asarray(image)
    grayScale = rgb2gray(dataset)
    binary = gray2binary(grayScale,100)
    inverseBinary = inverse(binary)
    erosed1 = erosion(binary)
    erosed2 = erosion(erosed1)
    generateImageFromMatrix(grayScale, "grayScale.png")
    generateImageFromMatrix(erosed2, "erosion2.png")
    generateImageFromMatrix(binary, "thresholding.png")

__main__()
