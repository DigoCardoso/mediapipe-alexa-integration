import cv2
import mediapipe as mp
import tensorflow as tf
#import tensorflow.keras as keras
from keras.models import load_model
import numpy as np
import json
import time


#funcao para escrita da letra traduzida no arquivo dados.json
def escrita_json(letra):
    dicionario = {
        "Letra":"{}".format(letra)
    }

    with open("dados.json","w") as file:
        json.dump(dicionario,file)
            

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(max_num_hands=1)

classes = ['A','B','C','D','Background']
model = load_model(filepath='keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

trad = 'N'
ult_trad = 'N'
atual = 'N'
cont_trad = 0
consist = False
envio = False
indexVal = 4

while True:
    success, img = cap.read()
    frameRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints != None:
        for hand in handsPoints:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            #cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

            try:
                imgCrop = img[y_min-50:y_max+50,x_min-50:x_max+50]
                imgCrop = cv2.resize(imgCrop,(224,224))
                imgArray = np.asarray(imgCrop)
                normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                data[0] = normalized_image_array
                prediction = model.predict(data)
                indexVal = np.argmax(prediction)
                #cv2.putText(img,classes[indexVal],(x_min-50,y_min-65),cv2.FONT_HERSHEY_COMPLEX,3,(0,0,255),5)

            except:
                continue
    
    trad = classes[indexVal]
    #print("Valor de trad = {}".format(trad))
    #print("Valor de ult_trad = {}".format(ult_trad))
    #print("Valor de cont_trad = {}".format(cont_trad))
    if trad != "Background":
        if (not consist):
            if trad == ult_trad:
                if cont_trad < 10:
                    cont_trad = cont_trad + 1
                else:
                    consist = True
                    atual = trad
            else:
                cont_trad = 0
        else:
            if trad == atual:
                if cont_trad < 10:
                    cont_trad = cont_trad + 1
            else:
                if cont_trad > 5:
                    cont_trad = cont_trad - 1
                else:
                    cont_trad = 0
                    consist = False
                    envio = False


    if consist and (not envio):
        escrita_json(atual)
        envio = True
        print("ESCRITA REALIZADA!!")


    ult_trad = trad

    cv2.imshow('Imagem',img)
    cv2.waitKey(1)
    #time.sleep(0.5) #sleep util para identificar os resultados mais devagar