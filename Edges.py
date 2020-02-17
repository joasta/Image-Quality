import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt

eoi = 0
ale = 1
ple = 2
nep = 3

def EBIQA(img):
    lap = cv.Laplacian(img,cv.CV_8UC1)
    ret,lap = cv.threshold(lap,127,255,cv.THRESH_BINARY)

    ex = (int(lap.shape[0]/16)+1)*16
    why = (int(lap.shape[1]/16)+1)*16
    #rozmiar podzielny na 16
    lap = cv.copyMakeBorder(lap, 0 , why-lap.shape[1], 0 , ex-lap.shape[0], cv.BORDER_CONSTANT, None, 0)

    X = int(lap.shape[0]/16)
    Y = int(lap.shape[1]/16)

    N = 16
    O = [[[None for _ in range(4)] for _ in range(Y)] for _ in range(X)]

    for i in range(X):
        for j in range(Y):
            lngth = 0
            x=i*N
            y=j*N
            mx_area = 0
            mx=[0,0,0,0]

            roi = lap[y:y+N, x:x+N]
            roi2 = lap[y:y+N, x:x+N]
            contours, hierarchy = cv.findContours(roi, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            for ele in contours:
                if len(ele) == 1 and lngth <1:
                    lngth = 1
                if len(ele) > 1:
                    le = []
                    for k in range(len(ele)-1):
                        tmp = ele[k]-ele[k+1]
                        le.append(np.linalg.norm(tmp))
                    lngth = sum(le) / len(le)

                x,y,w,h = cv.boundingRect(ele)
                area = w*h
                if area > mx_area:
                    mx = x,y,w,h
                    mx_area = area
            x,y,w,h = mx
            roi2=roi[y:y+h,x:x+w]

            whites = np.sum(roi == 255)
            whites2 = np.sum(roi2 == 255)

            O[i][j][nep] = whites
            O[i][j][ple] = whites2
            O[i][j][eoi] = len(contours)
            O[i][j][ale] = lngth
    return O

def EBIQA_p2(O, D):
    n = len(D)
    m = len(D[0])
    d = [[None for _ in range(m)] for _ in range(n)]
    suma = 0
    for i in range(n):
        for j in range(m):
            tmp = (O[i][j][eoi] - D[i][j][eoi])**2 + (O[i][j][ale] - D[i][j][ale])**2
            tmp += (O[i][j][ple] - D[i][j][ple])**2 + (O[i][j][nep] - D[i][j][nep])**2
            d[i][j] = sqrt(tmp)
            suma += sqrt(tmp)
    suma = suma/(n*m)
    return suma

if __name__ == "__main__":
    imges = ['bird.jpg', 'bird1.jpg', 'bird2.jpg', 'bird3.jpg', 'bird4.jpg', 'bird5.jpg']
    #imges = ['original.png', 'distorted.png', 'distorted2.png', 'distorted3.png']

    img = cv.imread(imges[0],0)
    O = EBIQA(img)
    for i in range(len(imges)-1):
        img_b = cv.imread(imges[i+1],0)
        D = EBIQA(img_b)
        print(EBIQA_p2(O, D))




#plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(1,2,2),plt.imshow(roi2,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
#plt.show()

