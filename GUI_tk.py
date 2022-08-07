import sys
import time
import tkinter as tk
import tkinter.messagebox as msg
import tkinter.font as tkFontS
from tkinter import *
from PIL import Image
from PIL import ImageTk
import cv2
import os
import imutils
# paquetes para el reconocimiento del lenguaje
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
categoriasT = []

DATA_PATH = os.path.join('DATA_SET')
data_Words = os.listdir(DATA_PATH)
actions = np.array(data_Words)
no_sequences = 20  # folder numbers
sequence_length = 20  # arrary numbers
label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

model = load_model('actions.h5')
model.load_weights('actions_weights.h5')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))


def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([rh, lh])


sentence = []
sequence = []
threshold = 0.7

win_init = tk.Tk()
win_init.title('Inicio')
bg_Principal = PhotoImage(file="comunicacion.png")
# defininr el anchoxalto
width_window = 800
height_window = 565
# funcion para centrar al inicio de la aplicacion
x_ventana = win_init.winfo_screenwidth() // 2 - width_window // 2
y_ventana = win_init.winfo_screenheight() // 2 - height_window // 2
position = str(width_window) + "x" + str(height_window) + "+" + str(x_ventana) + "+" + str(y_ventana)


def Camara():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # msg.showinfo('Informacion', message='Camara Abierta!!, pulse Aceptar', icon='info')
    while True:
        ret, frame = cap.read()
        if not ret:
            msg.showerror('Error', message='No se detecto la camara')
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Camara", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or not cv2.getWindowProperty("Camara", cv2.WND_PROP_VISIBLE):
            break

    cap.release()
    cv2.destroyAllWindows()


def Salir():
    win_init.destroy()
    sys.exit()


def Palabras():
    global categoriasT
    win_Words = tk.Toplevel()
    win_Words.title('Palabras')
    win_Words.geometry("400x280+500+250")
    win_Words.overrideredirect(False)
    win_Words.wm_attributes("-topmost", False)
    win_Words.wm_attributes("-disabled", False)
    win_Words.resizable(False, False)
    win_Words.grab_set()

    lbl_WordList = Label(win_Words, text="Palabras y Letras Registradas: ")
    lbl_WordList.place(x=10, y=10)

    if not os.path.exists("DATA_SET"):
        msg.showerror('Aviso', 'No se encontraron palabras registradas', icon='error')
        win_Words.destroy()
    else:
        categoriasT = os.listdir('DATA_SET')

    scrollbar2 = tk.Scrollbar(win_Words, orient=VERTICAL)
    lstWords2 = Listbox(win_Words, width=20, yscrollcommand=scrollbar2.set)
    scrollbar2.config(command=lstWords2.yview)
    scrollbar2.pack(ipadx=2, ipady=60, pady=(5, 40), side=RIGHT)

    scrollbar1 = tk.Scrollbar(win_Words, orient=VERTICAL)
    lstWords1 = Listbox(win_Words, width=20, yscrollcommand=scrollbar1.set)
    scrollbar1.config(command=lstWords1.yview)
    scrollbar1.pack(ipadx=2, ipady=60, pady=(5, 70), side=BOTTOM)

    for palabras in categoriasT:
        if len(palabras) > 1:
            if palabras == 'Fondo Vacio':
                palabras.lstrip('Fondo Vacio')
            else:
                lstWords1.insert(END, palabras)
                lstWords1.place(x=50, y=40)
        elif len(palabras) == 1:
            lstWords2.insert(END, palabras)
            lstWords2.place(x=220, y=40)

    btn_Back = Button(win_Words, text="Atras", command=win_Words.destroy)
    btn_Back.place(x=300, y=220)
    win_Words.mainloop()


def Start_Recognition():
    global sentence, sequence, threshold, res
    msg.showinfo('Ayuda', 'Para Cerrar la Camara presiona la tecla ESC')
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)
            sequence.insert(0, keypoints)
            sequence = sequence[:20]
            x1 = int(0.6 * image.shape[1])
            y1 = 200
            x2 = image.shape[1] - 30
            y2 = int(0.7 * image.shape[1])
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(214, 114, 109), thickness=2)

            if len(sequence) == 20:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]

            try:
                if res[np.argmax(res)].any() > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
            except IndexError:
                pass

            if len(sentence) > 1:
                sentence = sentence[-1:]

            cv2.rectangle(image, (0, 0), (640, 40), (225, 140, 25), -1)
            if sentence == ['Fondo Vacio']:
                cv2.putText(image, text=' ', org=(3, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
            else:
                cv2.putText(image, text=' '.join(sentence), org=(3, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
                # print(sentence)
                time.sleep(0.010)

            cv2.imshow('Signal Translator', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# insertar imagen de fondo
label = Label(win_init, image=bg_Principal, bg='white')
win_init.overrideredirect(False)
win_init.geometry(position)
win_init.wm_attributes("-topmost", False)
win_init.wm_attributes("-disabled", False)
win_init.resizable(False, False)
label.pack()
label.place(x=0, y=0)
# etiqueta
fontStyle = tkFontS.Font(family="Comic Sans MS", size=40)
etiq = tk.Label(win_init, text='Bienvenido', bg='white', height=1, fg='black', font=fontStyle)
etiq.pack()
etiq.place(x=60, y=20)
etiq = tk.Label(win_init, text='Usuario', bg='white', height=1, fg='black', font=fontStyle)
etiq.pack()
etiq.place(x=560, y=20)

barraMenu = Menu(win_init)  # crea la barra de menu
# se crea las opciones que tendra el menu, se modifica las opciones de menu, bg, fg, font, etc.
menuOptions = Menu(barraMenu, tearoff=0)
menuOptions.add_command(label='Probar Camara', command=Camara)
menuOptions.add_command(label='Palabras', command=Palabras)
menuOptions.add_separator()  # separador entre opciones
menuOptions.add_command(label='Salir', command=Salir)
barraMenu.add_cascade(label="Opciones", menu=menuOptions)  # modo de desplegado de las opciones
win_init.config(menu=barraMenu)  # se inicia la barra de menu en la ventana

BtnStart = Button(win_init, text="Empezar", font=tkFontS.Font(family='Times New Roman', size=14),
                  command=Start_Recognition)
BtnStart.place(x=360, y=240)
btnExit = Button(win_init, text='Salir', font=tkFontS.Font(family='Times New Roman', size=14), command=Salir)
btnExit.place(x=375, y=300)

win_init.mainloop()
