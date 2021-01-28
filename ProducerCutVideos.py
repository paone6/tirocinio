import cv2
import os
import numpy as np
import glob
import dlib
import csv
import math
import time
import shutil
import psutil

#ToDo
#1 Ristrutturare lo script in classi - Fatto
#2 Permettere la selezione di più di un video alla volta - Fatto
#3 Aggiungere l'estrazione dei frame che contengono 'labbra', 'occhi', ''naso invece del solo contorno del volto - Fatto
#4 Estrarre solo le 'labbra' dal video - fatto ma commentato
#5 Implementare la libreria dlib - Fatto (libreria machine learning)
#6 Ottenere individuare i landmark su ogni singolo frame - Fatto

#-------------------
#Questo script effettua un'estrapolazione dei frame in un video che contengono un volto, scartando tutti quei frame
#che non ne contengono. Restituscie in output un video contenente solo frame che hanno un volto.

#Path completo del video
completePath = ""

class EditVideo(object):

    directoryDestinationFrame = None
    directoryDestinationVideo = None
    directoryDestinationFullVideo = None
    frameRate = 0
    durataMaxSecVideo = 0
    numChunk = 0

    def __init__(self, whereSaveVideo, whereSaveFrame, frameRate = 24, groupOfContiguosFrame = 6, durataMaxSecVideo = 600):
        
        #Dove saranno salvati i singoli frame del video
        self.directoryDestinationFrame = whereSaveFrame
        
        #Dove sara' salvati i video prodotti dal programma
        self.directoryDestinationVideo = whereSaveVideo

        self.directoryDestinationFullVideo = "/Users/paone/Desktop/VideoDataset/frameVideo/fullVideo"
        #frameRate del video
        self.setFrameRate(frameRate)

        #frame contigui
        self.setContiguosFrames(groupOfContiguosFrame)

        #durata massima del chunk del video (default 10 min)
        self.durataMaxSecVideo = durataMaxSecVideo

    def setFrameRate(self, frameRate):
        self.frameRate = frameRate

    def setContiguosFrames(self, groupOfContiguosFrame):
        '''if (self.frameRate % groupOfContiguosFrame == 0):
            self.NUM_FRAME_CONTIGUOS = groupOfContiguosFrame'''
        if (self.frameRate == 30):
            self.NUM_FRAME_CONTIGUOS = 30 
        elif(self.frameRate == 25):
            self.NUM_FRAME_CONTIGUOS = 25 
        elif(self.frameRate == 24):
            self.NUM_FRAME_CONTIGUOS = 24

  
    def pictureCreate(self, sourceVideo, namePicture, extension, showImage = False, isResize = True, percentSizeFrame = 30, drawRectOnImage = False):

        isDrawRectangleActive = drawRectOnImage
        durataTotVideo = 0
        videoToEdit = cv2.VideoCapture(sourceVideo)

        arrayTempRow = [] #Contine valori che vsnno inseriti alla fine del file csv

        print("Provo ad aprire il file video: " + sourceVideo)
        if videoToEdit.isOpened() == False:
            return False

        #face_cascade = cv2.CascadeClassifier('/Users/francescorullo/opt/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
        
        eye_cascade = cv2.CascadeClassifier('/Users/paone/anaconda3/pkgs/libopencv-3.4.2-h7c891bd_1/share/OpenCV/haarcascades/haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier('/Users/paone/Desktop/OpenCV/haarcascade_mcs_mouth.xml')
        nose_cascade = cv2.CascadeClassifier('/Users/paone/Desktop/OpenCV/haarcascade_mcs_nose.xml')

        # imposto il predittore a 68 punti della libreria dlib
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("/Users/paone/Desktop/VideoDataset/shape_predictor_68_face_landmarks.dat")

        success,image = videoToEdit.read()
        #Frame Num = count
        count = 0
        prevCount = -1
        #Numero di frame effettivamente presi
        effectiveCountFrame = 0
        #Numero frame contigui
        effectiveCountFrameCountiguos = 0
        #Array di frame contigui
        arrayImageFrameCountiguos = []

        namePicture = namePicture.split('.')[0] + "_"
        self.frameRate = int(namePicture.split("_")[4])

        print("<-------------------------------- Video Aperto-------------------------------->")
        print("Frame rate: " + str(self.frameRate) + " Frame contigui: " + str(self.NUM_FRAME_CONTIGUOS))

        while success:

            os.chdir(self.directoryDestinationFrame)

            nameVideo = namePicture.split('.')[0]
            #arrayRowFileCSV.append(nameVideo) #1 Nome video
            #arrayRowFileCSV.append(self.frameRate) #2 FPS ATTENZIONE TALE INFORMAZIONE VA PRESA DAL NOME DEL VIDEO

            if isResize:
                if (percentSizeFrame == -1):
                    width = int(850)
                    height = int(500)
                else:
                    width = int(image.shape[1] * percentSizeFrame / 100)
                    height = int(image.shape[0] * percentSizeFrame / 100)
                dim = (width, height)
                image = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
                imageFullVideo = image.copy()

            if effectiveCountFrame == self.frameRate:

                durataTotVideo = durataTotVideo + 1
                effectiveCountFrame = 0
                print("Durata video: " + str(durataTotVideo))
                if durataTotVideo % self.durataMaxSecVideo == 0:
                    print("Durata Massima raggiunta, avvio terminazione...")
                    editVideo.videoCreate(".jpg", namePicture + str(self.numChunk), self.directoryDestinationVideo)
                    editVideo.picturesDelete("/Users/paone/Desktop/VideoDataset/frameVideo", ".jpg")
                    self.numChunk = self.numChunk + 1
                    os.chdir(self.directoryDestinationFrame)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            #Volto
            #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            faces_dlib = detector(gray, 0) #dlib detector

            #if len(faces) == 0 and os.path.exists(namePicture + "%d" % count + extension):
                    #self.frameDelete(namePicture + "%d" % count + extension)
                    #effectiveCountFrame -= 1

            isFaceVisible = False
            for face in faces_dlib:

                (x, y, h, w) = self.rect_to_bb(face)

                if isDrawRectangleActive:
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

                roi_gray = gray[y:y+h, x:x+w]
                #roi_color = image[y:y+h, x:x+w]

                # assegno i landmark
                landmarks = predictor(gray, face)

                # inserisco i landmark in una lista 
                landmarks_list = []
                for i in range(0, landmarks.num_parts):
                    landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

                if (len(landmarks_list) > 0):
                    arrayTempRow.append(str(len(landmarks_list)))#5 Numero di Landmarks
                    
                    arrayCoordinate = []
                    for landmark_num, xy in enumerate(landmarks_list, start=1):
                        arrayCoordinate.append((xy[0], xy[1]))
                    arrayTempRow.append(arrayCoordinate) #6 Cooridnate dei landmark (vengono inseriti infondo questo è un array temporaneo)

                    #Caloclo la distanza dei landmark con la strategia 1 a tutti
                    arrayDistanzePunti = []
                    for landmark_numX, xy in enumerate(landmarks_list, start=1):
                        coordinatePivotX = xy[0]
                        coordinatePivotY = xy[1]
                        for landmark_numY, xy in enumerate(landmarks_list, start = landmark_numX + 1):
                            coordinatePointX = xy[0]
                            coordinatePointY = xy[1]
                            #calcolo distanza euclidea
                            distanza = math.sqrt((coordinatePivotX - coordinatePointX)**2 + (coordinatePivotY - coordinatePointY)**2)
                            arrayDistanzePunti.append(str(distanza) + ",")
                    arrayTempRow.append(arrayDistanzePunti) #7 Distanza punti 1 a tutti
                            

                #self.distanceLandmarks(landmarks_list)

                if isDrawRectangleActive:
                    for landmark_num, xy in enumerate(landmarks_list, start = 1):
                        cv2.circle(image, (xy[0], xy[1]), 12, (168, 0, 20), -1)
                        cv2.putText(image, str(landmark_num),(xy[0]-7,xy[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)

                #Occhi
                eyes = eye_cascade.detectMultiScale(roi_gray)

                if isDrawRectangleActive:
                    for (ex,ey,ew,eh) in eyes:
                        cv2.rectangle(image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                
                #Naso
                nose = nose_cascade.detectMultiScale(gray, 1.3, 5)
                
                if isDrawRectangleActive:
                    for (x,y,w,h) in nose:
                        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)

                #Bocca
                mouth = mouth_cascade.detectMultiScale(gray, 1.7, 11)
                
                for (x,y,w,h) in mouth:
                    y = int(y - 0.15 * h)
                    #imageMouth = image[y:y+h, x:x+w]
                    if isDrawRectangleActive:
                        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 3)

                if  len(eyes) < 2 or len(mouth) == 0 or len(nose) == 0: #(len(landmarks_list) <= 0):
                    isFaceVisible = False
                    effectiveCountFrameCountiguos = 0
                    arrayImageFrameCountiguos.clear()
                    if os.path.exists(namePicture + "%d" % count + extension):
                        self.frameDelete(namePicture + "%d" % count + extension)
                    
                else:
                    #Verifico se e' contiguo
                    if count - prevCount == 1 or len(arrayImageFrameCountiguos) == 0:
                        arrayImageFrameCountiguos.append(image)
                        effectiveCountFrameCountiguos +=1
                    else:
                        arrayImageFrameCountiguos.clear()
                        effectiveCountFrameCountiguos = 0
                    prevCount = count
                    isFaceVisible = True
                        
                if effectiveCountFrameCountiguos == self.NUM_FRAME_CONTIGUOS:
                    internalCount = count - self.NUM_FRAME_CONTIGUOS
                    for img in arrayImageFrameCountiguos:
                        cv2.imwrite(namePicture + "%d" % internalCount + extension, img)
                        internalCount += 1
                    effectiveCountFrame += self.NUM_FRAME_CONTIGUOS
                    effectiveCountFrameCountiguos = 0
                    arrayImageFrameCountiguos.clear()

            '''if (isFaceVisible):
                arrayRowFileCSV.append("SI") #3 e' presente un volto
            else:
                arrayRowFileCSV.append("NO") #3 e' presente un volto

            arrayRowFileCSV.append(str(count)) #4 Frame corrente
            '''
            os.chdir(self.directoryDestinationFullVideo)
            if (isFaceVisible):
                cv2.circle(imageFullVideo, (15, 15), 40, (0, 255, 0), -1)
                cv2.putText(imageFullVideo, str(count),(15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)
            else:
                cv2.circle(imageFullVideo, (15, 15), 40, (0, 0, 255), -1)
                cv2.putText(imageFullVideo, str(count),(15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255), 1)

            cv2.imwrite(namePicture + "%d" % count + extension, imageFullVideo)
            count += 1
            '''
            if (len(arrayTempRow) > 1):
                arrayRowFileCSV.append(arrayTempRow[0]) #5Num di lanmarks
                arrayRowFileCSV.append(arrayTempRow[1]) #6 Coordiante dei landmarks
                arrayRowFileCSV.append(arrayTempRow[2]) #7 distanza punti 1 a tutti separati con ,
            else:
                arrayRowFileCSV.append("0") #5 Num di landmarks

            arrayGlobalFileCSV.append(arrayRowFileCSV.copy())
            '''
            success,image = videoToEdit.read()
            '''
            editVideo.openCSVFile(self.pathCSV, "a", ",", arrayGlobalFileCSV)
            arrayRowFileCSV.clear()
            arrayGlobalFileCSV.clear()
            arrayTempRow.clear()
            '''
            if showImage:
                cv2.imshow('image',image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            #print('Nuovo Frame: ', success, count)

        return True

    #calcolo distanza tra i landmarks
    #def distanceLandmarks(self, listOfLandmarks):
    #    for landmark_num, xy in enumerate(listOfLandmarks, start = 1):


    #Cancella uno specifico frame
    def frameDelete(self, nameFrame):
        os.remove(nameFrame)

    #Cancella tutti i frame creati nel path specificato
    def picturesDelete(self, path, extension):

        for filename in glob.glob(path + '/*' + extension):
            os.remove(filename)

    #Assembla i frame creati dalla procedura pictureCrate creando un video
    def videoCreate(self, extensionFrame, nameVideo, directoryVideoSaved):

        os.chdir(directoryVideoSaved)

        image_array = []
        for filename in sorted(glob.glob(self.directoryDestinationFrame + '/*' + extensionFrame), key=os.path.getmtime):
            
            #print("---> " + filename)
            image = cv2.imread(filename)
            height, width, layers = image.shape
            size = (width,height)
            image_array.append(image)
        
        print("<--- Numero Frame inseriti nel video: " + str(len(image_array)) + "--->")

        if len(image_array) == 0:
            return False

        else:
            videoName = nameVideo + ".avi"
            out = cv2.VideoWriter(videoName,cv2.VideoWriter_fourcc(*'DIVX'), self.frameRate, size)
        
            for i in range(len(image_array)):
                out.write(image_array[i])

            out.release()
            
            return True
    #calcolo distanza tra i landmarks
    #def distanceLandmarks(self, listOfLandmarks):
    #    for landmark_num, xy in enumerate(listOfLandmarks, start = 1):


    #Cancella uno specifico frame
    def frameDelete(self, nameFrame):
        os.remove(nameFrame)

    #Cancella tutti i frame creati nel path specificato
    def picturesDelete(self, path, extension):

        for filename in glob.glob(path + '/*' + extension):
            os.remove(filename)

   
    #Assembla i frame creati dalla procedura pictureCrate creando un video
    def videoFullCreate(self, extensionFrame, nameVideo, directoryVideoSaved):

        os.chdir(self.directoryDestinationVideo)

        #image_array = []
        videoName = nameVideo + ".avi"
        out = cv2.VideoWriter(videoName,cv2.VideoWriter_fourcc(*'DIVX'), self.frameRate, (0, 0))
        numFrame = 0
        for filename in sorted(glob.glob(self.directoryDestinationVideo + '/*' + extensionFrame), key=os.path.getmtime):
            
            #print("---> " + filename)
            image = cv2.imread(filename)
            height, width, layers = image.shape
            size = (width,height)
            out.size = size
            out.write(image)
            #image_array.append(image)

            numFrame += 1

            memory = psutil.virtual_memory()
            print("Memory Used: " + memory[2])
        
        out.release()
        
        print("<--- Numero Frame inseriti nel video: " + str(numFrame) + "--->")
            
        return True

    #converte l'oggetto restituito dalla dlib dopo aver identificato un volto
    #e lo converte nelle coordinate x,y,h,w per poter lavorare con la bounding box
    def rect_to_bb(self, rect):
        
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        return (x, y, w, h)
    
    

print("Start...")

editVideo = EditVideo("/Users/paone/Desktop/VideoDataset/frameVideo/Video",  "/Users/paone/Desktop/VideoDataset/frameVideo", 30, 3, 15)

path = "/Users/paone/Desktop/Spagnolo/Uomo/Over_30" #path della cartella contenente i video
numVideo = 0
nameVideo = ""


for fileVideo in os.listdir(path):
                                print("trovato video")
                                nameVideo = fileVideo
                                print(nameVideo)
                                if (len(nameVideo.split('_')) >= 4):
                                    frame = nameVideo.split('_')[4]
                                    print(frame)
                                    frame = frame.split('.')[0]
                                    print(frame)
                                    editVideo.setFrameRate(int(frame))
                                    editVideo.setContiguosFrames(3)
                                
                                #editVideo.picturesDelete("/Users/paone/Desktop/VideoDataset/frameVideo", ".jpg")
                                completePathVideo = path + "/" + fileVideo
                                print("Video: " + completePathVideo)
                                isSuccess = editVideo.pictureCreate(completePathVideo, nameVideo, ".jpg", False, True, -1, False)
                                #isVideoCreated = editVideo.videoCreate(".jpg", nameVideo + str(numVideo)) #Creo un video con i frame rimanenti
                                print("Fine Video num: " + str(numVideo) + " - " + nameVideo)
                                if isSuccess:
                                    editVideo.videoFullCreate(".jpg", nameVideo, editVideo.directoryDestinationVideo)
                                    #shutil.move(completePathVideo, "/Users/paone/Desktop/Video_Computati/" + nameVideo)
                                    numVideo += 1
                                editVideo.picturesDelete("/Users/paone/Desktop/videoDataset/frameVideo", ".jpg")
                                editVideo.picturesDelete("/Users/paone/Desktop/VideoDataset/frameVideo/fullVideo", ".jpg")

     
                
editVideo.picturesDelete("/Users/paone/Desktop/videoDataset/frameVideo", ".jpg")
editVideo.picturesDelete("/Users/paone/Desktop/VideoDataset/frameVideo/fullVideo", ".jpg")

print("Fine")