#Deklarasi import library yang digunakan
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI_TA_A1.ui', self)
        #Deklarasi setiap trigger ataupun button
        self.Image = None
        self.actionOpen.triggered.connect(self.openClicked)
        self.actionSave.triggered.connect(self.save)
        self.button_Face.clicked.connect(self.face)
        self.actionGrayscalling.triggered.connect(self.grayscale)
        self.actionGaussian_Filter.triggered.connect(self.filter)

    def grayscale(self):
        # Deklarasi variable panjang dan lebar pada array 2 dimensi
        # yang menyimpan panjang dan lebar citra
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        #Perulangan pada panjang dan lebar disetiap pixel citra
        for i in range(H):
            for j in range(W):
                #Proses grayscaling
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        # Output Gambar
        self.Image = gray
        self.displayImage(2)
        print('Nilai Pixel Grayscale :',self.Image[0:1, 0:1])

    def openClicked(self):
        flname, filter = QFileDialog.getOpenFileName(self, 'Open File', 'C:\\User\\')
        if flname:
            self.loadImage(flname)
        else:
            print('Invalid Image')

    def save(self):
        flname, filter = QFileDialog.getSaveFileName(self, 'save file', 'G:\\', "Images Files(*.jpg)")
        if flname:
            cv2.imwrite(flname, self.Image)
        else:
            print('Saved')

    def face(self):
        gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
        mouth_test = mouth_cascade.detectMultiScale(gray, 1.7, 11)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        H, W = self.Image.shape[:2]
        for (x, y, w, h) in faces:
            cv2.rectangle(self.Image, (x, y), (x + w, y + h), (0, 0, 0), 8)
        if (len(faces) >= 1):
            if (len(mouth_test) == 0):
                cv2.putText(self.Image, "Mask", (faces[0][0], faces[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                self.label_Status.setText(str("Menggunakan Masker"))
            elif (len(mouth_test) >= 1):
                cv2.putText(self.Image, "No Mask", (faces[0][0], faces[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        3)
                self.label_Status.setText(str("Tidak Menggunakan Masker"))
        if (len(faces) == 0):
            cv2.putText(self.Image, "NoWajah", (W // 2, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
            self.label_Status.setText(str("Wajah Tidak Terdeteksi"))
        print(faces)
        print(mouth_test)
        self.displayImage(2)

    def convolve(self, img, kernel):
        img_height = img.shape[0]
        img_width = img.shape[1]
        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]
        H = (kernel_height) // 2
        W = (kernel_width) // 2
        out = np.zeros((img_height, img_width))
        for i in np.arange(H + 1, img_height - H):
            for j in np.arange(W + 1, img_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = img[i + k, j + l]
                        w = kernel[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum
        return out

    def filter(self):
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        gaussian = (1.0 / 345) * np.array(
           [[1, 5, 7, 5, 1],
            [5, 20, 33, 20, 5],
            [7, 33, 55, 33, 7],
            [5, 20, 33, 20, 5],
            [1, 5, 7, 5, 1]])
        self.Image = self.convolve(img, gaussian)
        plt.imshow(self.Image, cmap='gray', interpolation='bicubic')
        plt.show()

    def loadImage(self,flname):
        self.Image=cv2.imread(flname)
        self.displayImage(1)
        self.Image2=self.Image

    def displayImage(self, windows):
        qformat = QImage.Format_Indexed8
        if len(self.Image.shape)==3:
            if(self.Image.shape[2])==4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0],
                     self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows ==1:
            self.label_Citra.setPixmap(QPixmap.fromImage(img))
            self.label_Citra.setScaledContents(True)
        if windows ==2:
            self.label_Proses.setPixmap(QPixmap.fromImage(img))
            self.label_Proses.setScaledContents(True)


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle ('Aplikasi Pengolahaan Citra')
window.show()
sys.exit(app.exec())