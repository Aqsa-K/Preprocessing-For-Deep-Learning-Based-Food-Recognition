import cv2
import numpy as np

def histogram_equalize(img):
    b, g, r = cv2.split(img)                    #split image into bgr channels (blue, green, red - this is the default order for function cv2.split)
    red = cv2.equalizeHist(r)                   #carry out histogram equalizaton for each channel
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((blue, green, red))        #merge back the three channels

def grey_world(img):
    alpha , beta, gamma = 1,1,1

    b, g, r = cv2.split(img)                    # split into red green and blue channels
    sumB = np.sum(b)
    sumG = np.sum(g)
    sumR = np.sum(r)
    n = np.ma.size(b)                           # no of elements - will be same for all channels

    #calculate averages
    avgB = (sumB/n)
    avgG = (sumG/n)
    avgR = (sumR/n)

    scale = float((avgB + avgG + avgR))/3.0

    denB = float(avgB) * gamma
    denG = float(avgG) * beta
    denR = float(avgR) * alpha

    blue = np.divide((np.multiply(b,scale)), denB)
    green = np.divide((np.multiply(g,scale)), denG)
    red = np.divide((np.multiply(r,scale)), denR)

    blue = np.array(blue, dtype= np.uint8)
    green = np.array(green, dtype= np.uint8)
    red = np.array(red, dtype= np.uint8)

    merged = cv2.merge((blue,green,red))
    return merged


if __name__ == "__main__":
    img = cv2.imread("Food_Item.jpg")

    #perform grey world on the image
    Img_Grey = grey_world(img)
    cv2.imshow("Food_Item - GreyWorld.jpg", Img_Grey)

    #now perform histogram_equalization on the resultant image
    Img_hist = histogram_equalize(Img_Grey)
    cv2.imshow("Food_Item - Histogram Equalized", Img_hist)


    cv2.waitKey(0)                      #the program will keep running until you enter a key