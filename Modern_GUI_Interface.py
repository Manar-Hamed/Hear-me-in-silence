from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from keras.models import load_model
import speech_recognition as sr
import customtkinter as ctk
import tensorflow as tf
from PIL import Image
from gtts import gTTS
import pandas as pd
import numpy as np
import random
import pygame
import cv2
import os
from pydub import AudioSegment



ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure the Window
        self.title("ArSL Translator")
        self.geometry(f"{1100}x{580}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Title
        self.label0 = ctk.CTkLabel(self, text="ArSL Transaltor", font=ctk.CTkFont(size=20, weight="bold"))
        self.label0.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # Left Side Frame
        self.lframe = ctk.CTkFrame(self, corner_radius=0)
        self.lframe.grid(row=1, column=0, rowspan=4, sticky="nsew")
        self.lframe.grid_rowconfigure(4, weight=1)

        # Input Type Label
        self.InputLabel = ctk.CTkLabel(self.lframe, text="Input Type", font=ctk.CTkFont(size=16, weight="bold"))
        self.InputLabel.grid(row=3, column=0, padx=(20, 0), pady=(20, 20), sticky="nsew")

        # Input Type Option Menu
        self.input_optionemenu = ctk.CTkOptionMenu(self.lframe, values=["Microphone", "Camera"], command=self.inputType)
        self.input_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 10))
        self.input_optionemenu.set("Microphone")  # Set Microphone as default

        # Right Side Frame
        self.rframe = ctk.CTkFrame(self, corner_radius=0)
        self.rframe.grid(row=1, column=1, rowspan=4, columnspan=3, sticky="nsew")

        # Image Objects
        micPhoto = ctk.CTkImage(dark_image= Image.open("gui_icons\microphone_icon.png"), size=(80,80))
        camPhoto = ctk.CTkImage(dark_image= Image.open("gui_icons\camera_icon.png"), size=(80,80))
        playPhoto = ctk.CTkImage(dark_image=Image.open("gui_icons\play_icon.png"), size=(80,80))

        # MIC Button
        self.micButton = ctk.CTkButton(self.rframe, text="", image=micPhoto, command=self.mic, fg_color="transparent")
        self.micButton.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))

        # CAM Button
        self.camButton = ctk.CTkButton(self.rframe, text="", image=camPhoto, command=self.cam, fg_color="transparent")

        # Browse Button
        self.browseButton = ctk.CTkButton(self.rframe, text="Browse", command=self.browse)
        self.browseButton.grid(row=1, column=2, padx=(20, 20), pady=(20, 20))
        

        # File Path Entry
        self.fileEntry = ctk.CTkEntry(self.rframe, placeholder_text="File Path")
        self.fileEntry.grid(row=1, column=0, columnspan=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # Text Label
        self.textLabel = ctk.CTkLabel(master=self.rframe, text="Click on the Mic Button and Speak", font=ctk.CTkFont(size=18, weight="bold"), anchor="e")
        self.textLabel.grid(row=0, column=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        
        # Inner Frame
        self.inframe = ctk.CTkFrame(master=self.rframe, corner_radius=0)
        self.inframe.grid(row=2, column=0, sticky="nsew")

        # Images Canvas
        self.imgCanvas = ctk.CTkCanvas(master=self.inframe)
        self.imgCanvas.grid(row=2, column=0, sticky="nsew")

        # Scroll Window
        self.vscroll = ctk.CTkScrollbar(master=self.inframe, command=self.imgCanvas.yview)
        self.vscroll.grid(row=2, column=4, sticky="ns")

        # Configuration
        self.imgCanvas.configure(yscrollcommand=self.vscroll.set)

        # Play Button
        self.playButton = ctk.CTkButton(self.rframe, text="", image=playPhoto, command=self.play_audio, fg_color="white")

    # Defaults/Globals
    guide = pd.read_csv('data\Arabic_Letters_Guide.csv') # Loading the Arabic Letters Guide
    imgdir = 'data\ArASL_Database_54K_Final\ArASL_Database_54K_Final'
    special_characters = ['@', '!', '؟', '$', '%', '^', '*', '-', '_']
    ArSL = []
    mode = "Microphone"

    # Commands Section
    def inputType(self, input_mode: str):

        self.mode = input_mode
    
        if input_mode == "Microphone":
            self.camButton.grid_forget()
            # self.browseButton.grid_forget()
            self.fileEntry.delete(0, ctk.END)
            self.playButton.grid_forget()
            self.textLabel.configure(text="Click on the Mic Button and Speak")
            self.inframe.grid(row=2, column=0, rowspan=3, columnspan=4, sticky="nsew")
            self.micButton.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))
        else:
            self.fileEntry.delete(0, ctk.END)
            self.micButton.grid_forget()
            self.textLabel.configure(text="")
            self.inframe.grid_forget()
            self.camButton.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))
           


    def browse(self):  # Open Image Files
        tf = ctk.filedialog.askopenfilenames(
            initialdir=r".\data\\Test",
            title="Select Images",
            filetypes=(("All Files", "*.*"), ("Images", "*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG", ))
            )
        self.fileEntry.insert(ctk.END, tf)
        

        if self.mode == "Microphone":
            self.mp3_text(list(tf))
        else:
            self.img_speech(list(tf))
    
    imgs = []

    def img_speech(self, fp):
        # Loading the Model
        ArSL_model = load_model('models/ArSLText.h5')

        for name in fp:
            img = cv2.imread(name)
            # self.imgs.append(img)
            resize = tf.image.resize(img, (256, 256))
            np.expand_dims(resize, 0)
            yhat = ArSL_model.predict(np.expand_dims(resize/255, 0))
            result = np.where(yhat[0] == np.amax(yhat[0]))

            letter = self.guide[self.guide["Index"] == result[0][0]]["Arabic_Letters"].iloc[0]

            self.string += letter
        
        self.textLabel.configure(text=self.string)
        self.text_speech(self.string)  # Convert Text to Speech/Audio
        self.playButton.grid(row=4, column=0, padx=(20, 20), pady=(20, 20))



    def image_list(self):
        for cla in os.listdir(self.imgdir):
            p = self.imgdir + '\\' + cla
            imgName = random.choice(os.listdir(p))
            imPath = p + '\\' + imgName
            image = cv2.imread(imPath)
            self.ArSL.append(image)

    label = ""

    def mic(self): # Microphone Input
        r = sr.Recognizer()
        mic = sr.Microphone()

        # Using Google Cloud API
        with mic as audio_file:

            r.adjust_for_ambient_noise(audio_file)
            audio = r.listen(audio_file)

            try:
                text = r.recognize_google(audio, language='ar-EG')

                self.label = self.label + "\n" + text + " :لقد قلت"
                self.speect_text_image(text)

            except Exception as e:
                self.label = self.label + "\n" + 'Error: ' + str(e)

            self.textLabel.configure(text=self.label)
        
    def mp3_text(self, file_audio):
        file_audio = sr.AudioFile(file_audio[0])

        # use the audio file as the audio source                                        
        r = sr.Recognizer()
        with file_audio as source:
            audio_text = r.record(source)

        # print(type(audio_text))
        text = r.recognize_google(audio_text, language='ar-EG')
        self.speect_text_image(text)


    def speect_text_image(self, sentence):      

        # Special Characters Removal
        for sp in self.special_characters:
            sentence = sentence.replace(sp, '')
        
        words = sentence.split(' ')
        encoded = []

        for word in words:
            list_code = []
            for letter in word:
                code = self.guide[self.guide['Arabic_Letters'] == letter]['Index'].iloc[0]
                list_code.append(code)
            encoded.append(list_code)
            
        self.image_list()  # Fetch an image for each letter    
        
        # Displaying the Images
        for l1 in encoded:
            self.fig = Figure(figsize=(15,3), frameon=False)
            l1.reverse()
            self.ax = self.fig.subplots(1, len(l1))
            for i in range(len(l1)):
                self.ax[i].imshow(self.ArSL[l1[i]], cmap='gray')
                self.ax[i].set_title(self.guide[self.guide['Index'] == l1[i]]['Arabic_Letters'].iloc[0])
                self.ax[i].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.imgCanvas)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid(sticky="nsew")

    string = ""

    def cam(self):  # Camera Input
        # self.imageLabel.grid_forget()
        
        source = cv2.VideoCapture(0)
        color_dict = (0,255,0)
        count = 0
        prev_val = 0
        letter = ""
        self.string = ""

        ArSL_model = load_model('models/ArSLText.h5')

        while(True):
            ret, img = source.read()
            cv2.rectangle(img, (24,24), (350 , 350), color_dict, 2)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            crop_img = bgr[24:350,24:350]
            
            count += 1
            if (count % 100 == 0):
                prev_val = count
            
            cv2.putText(img, str(prev_val//100), (400, 150),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)

            resize = tf.image.resize(crop_img,(256,256))  # Resize to (256,256)
            np.expand_dims(resize, 0)
            normalized = resize/255  # Scale Image
            # gray = tf.image.rgb_to_grayscale(resize)
 
            yhat = ArSL_model.predict(np.expand_dims(normalized, 0))
            result = np.where(yhat[0] == np.amax(yhat[0]))

            if (count == 300):
                count = 99
                letter = self.guide[self.guide['Index'] == result[0][0]]['Arabic_Letters'].iloc[0]
                self.string += letter

            cv2.putText(img, letter, (24, 14),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) 
            cv2.putText(img, self.string, (357, 50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)
            
            cv2.imshow('LIVE',img)
            key = cv2.waitKey(1)
        
            if (key == 27):  # Press Esc to exit
                break
            
        cv2.destroyAllWindows()
        source.release()

        cv2.destroyAllWindows()

        self.textLabel.configure(text=self.string)
        self.text_speech(self.string)  # Convert Text to Speech/Audio
        self.playButton.grid(row=4, column=0, padx=(20, 20), pady=(20, 20))

    
    def text_speech(self, Text):

        txt_sound = gTTS(text=Text, lang='ar', slow=False)
        self.audioPath = 'audio/ArSL_Audio_' + Text + '.mp3'
        txt_sound.save(self.audioPath)
    

    pygame.mixer.init()  # initialise the pygame

    def play_audio(self):  # Using pygame
        pygame.mixer.music.load(self.audioPath)
        pygame.mixer.music.play(loops=0)

    # ----------------------------------------------------------------------


if __name__ == "__main__":
    app = App()
    app.mainloop()
