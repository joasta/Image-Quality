import cv2 as cv
import numpy as np
from math import sqrt

#indeksowanie tablicy EBIQA
eoi = 0
ale = 1
ple = 2
nep = 3

def EBIQA(img):
    """Funkcja obliczająca macierz parametrów podanego obrazu:
    EOI - Edge Orientation in Image
    ALE - Average Length of the Edges
    PLE - Primitive Length of the Edges
    NEP - Number of Edge Pixels
    Funkcja oblicza także uśredniony laplasjan.

    Parametry:
    img - obraz formatu OpenCV

    Funkcja zwraca:
    O - macierz parametrów N/16 x M/16 x 4 wymiarową (N, M - wymiary img)
    edge - uśredniona wartość laplasjanu wykonanego na img
    """

    #filtracja: laplasjan
    lap = cv.Laplacian(img,cv.CV_8UC1)
    #uśredniona wartość laplasjanu
    edge = cv.mean(lap)[0]

    #binaryzacja
    ret,lap = cv.threshold(lap,127,255,cv.THRESH_BINARY)

    #zmiana rozmiaru przefiltrowanego obrazu na podzielny przez 16
    ex = (int(lap.shape[0]/16)+1)*16
    why = (int(lap.shape[1]/16)+1)*16
    lap = cv.copyMakeBorder(lap, 0 , why-lap.shape[1], 0 , ex-lap.shape[0], cv.BORDER_CONSTANT, None, 0)

    X = int(lap.shape[0]/16)
    Y = int(lap.shape[1]/16)

    N = 16
    #inicjalizacja macierzy parametrów
    O = [[[None for _ in range(4)] for _ in range(Y)] for _ in range(X)]

    for i in range(X):
        for j in range(Y):
            lngth = 0
            x=i*N
            y=j*N
            mx_area = 0
            mx=[0,0,0,0]

            #wybór obszarów o rozmiarze 16x16px
            roi = lap[y:y+N, x:x+N]
            roi2 = lap[y:y+N, x:x+N]
            #wybór obszarów 
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

            #number of edge pixels
            O[i][j][nep] = whites
            #primitive length of the edges (piksele najdłuższej pojedynczej krawędzi)
            O[i][j][ple] = whites2
            #edge orientation in image (liczba krawędzi w obszarze)
            O[i][j][eoi] = len(contours)
            #average length of the edges (średnia długość konturów krawędzi)
            O[i][j][ale] = lngth
    return O, edge

def EBIQA_p2(O, D):
    """Funkcja obliczająca różnice macierzy parametrów:
    O - obrazu zarejestrowanego
    D - obrazu idealnego

    Funkcja zwraca:
    suma - znormalizowana suma odchyleń parametrów O od idealnych parametrów D
    """

    n = len(D)
    m = len(D[0])
    d = [[None for _ in range(m)] for _ in range(n)]
    suma = 0
    for i in range(n):
        for j in range(m):
            #suma kwadratów różnic parametrów eoi, ale, ple i nep (badanych i idealnych)
            tmp = (O[i][j][eoi] - D[i][j][eoi])**2 + (O[i][j][ale] - D[i][j][ale])**2
            tmp += (O[i][j][ple] - D[i][j][ple])**2 + (O[i][j][nep] - D[i][j][nep])**2
            #pierwiastek z tej sumy
            d[i][j] = sqrt(tmp)
            #dodanie do sumy odchyleń
            suma += sqrt(tmp)
    #normalizacja sumy odchyleń
    suma = suma/(n*m)
    return suma

if __name__ == "__main__":    
    """Funkcja porównująca obrazy rzeczywiste z idealnym.

    imges - lista nazw obrazów, zaczynając od idealnego; obrazy
        muszą znajdować się w folderze projektu

    Funkcja oblicza i wyświetla wyniki alrogymtu EBIQA i uśredniony laplasjan.
    """

    #imges = ['bird.jpg', 'bird1.jpg', 'bird2.jpg', 'bird3.jpg', 'bird4.jpg', 'bird5.jpg']
    #imges = ['original.png', 'distorted.png', 'distorted2.png', 'distorted3.png']
    imges = ['baloons.png', 'blurred.png', 'dots.png', 'blur-dots.png']

    img = cv.imread(imges[0],0)
    O, edge0 = EBIQA(img)
    for i in range(len(imges)-1):
        img_b = cv.imread(imges[i+1],0)
        D, edge = EBIQA(img_b)
        print(f'Plik {imges[i+1]}\nOstrość krawędzi oryginalnych: {edge0}\nOstrość nowego obrazu: {edge}\nWynik EBIQA: {EBIQA_p2(O, D)}\n----')