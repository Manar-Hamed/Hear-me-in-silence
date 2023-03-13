from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from bidi.algorithm import get_display

from matplotlib.figure import Figure

from arabic_reshaper import reshape

from keras.models import load_model

from googletrans import Translator

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



ctk.set_appearance_mode("dark")

ctk.set_default_color_theme("dark-blue")



class App(ctk.CTk):

    def __init__(self):

        super().__init__()


        # Configure the Window

        self.title("Mutli SLs Translator")


        #getting screen width and height of display

        width= self.winfo_screenwidth()

        height= self.winfo_screenheight()

        self.geometry("%dx%d" % (width, height))


        # configure grid layout (4x4)

        self.grid_columnconfigure(1, weight=1)

        self.grid_columnconfigure((2, 3), weight=0)

        self.grid_rowconfigure((0, 1, 2), weight=1)


        # Title

        self.label0 = ctk.CTkLabel(self, text="Multi Sign Languages Transaltor", font=ctk.CTkFont(size=18, weight="bold"), anchor=ctk.CENTER)

        self.label0.grid(row=0, column=1, columnspan=4)#, padx=(20, 0), pady=(20, 0), sticky="nsew")


        # Left Side Frame

        self.lframe = ctk.CTkFrame(self, corner_radius=0, fg_color='transparent')

        self.lframe.grid(row=1, column=0, rowspan=4, sticky="nsew")

        self.lframe.grid_rowconfigure(4, weight=1)


        # Input Type Label

        self.InputLabel = ctk.CTkLabel(self.lframe, text="Input Type", font=ctk.CTkFont(size=14))#, weight="bold"))

        self.InputLabel.grid(row=1, column=0, padx=(10, 0), pady=(10, 0), sticky="nsew")


        # Input Type Option Menu

        self.input_optionemenu = ctk.CTkOptionMenu(self.lframe, values=["Microphone", "Camera", "Text"], command=self.inputType)

        self.input_optionemenu.grid(row=2, column=0, padx=10, pady=(10, 0))

        self.input_optionemenu.set("Microphone")  # Set Microphone as default


        # Source Language Label

        self.srcLangLabel = ctk.CTkLabel(self.lframe, text="From", font=ctk.CTkFont(size=14))


        # Source Language Option Menu

        self.srcLang_optionemenu = ctk.CTkOptionMenu(self.lframe, values=self.languages, command=self.fromlang)


        # Destination Language Label

        self.destLangLabel = ctk.CTkLabel(self.lframe, text="To", font=ctk.CTkFont(size=14))


        # Destination Language Option Menu

        self.destLang_optionemenu = ctk.CTkOptionMenu(self.lframe, values=self.languages, command=self.tolang)


        # Translator Status Label

        self.tStatusLabel = ctk.CTkLabel(self.lframe, text="Translator: OFF", text_color='red', font=ctk.CTkFont(size=12))

        self.tStatusLabel.grid(row=7, column=0)


        # Translator Mode Button

        # self.tranButton = ctk.CTkButton(self.lframe, text="Translator", command=self.tranMode)

        # self.tranButton.grid(row=8, column=0, padx=(20, 20), pady=(20, 20))

        # self.tranButton.configure(state='disabled')


        # Switch Variable

        self.switch_var = ctk.StringVar(value='OFF')


        # Translator Switch

        self.transSwitch = ctk.CTkSwitch(self.lframe, text='Translator', command=self.tranMode, variable=self.switch_var, onvalue='ON', offvalue='OFF')

        self.transSwitch.grid(row=8, column=0, padx=(20, 20), pady=(20, 20))


        # Right Side Frame

        self.rframe = ctk.CTkFrame(self, corner_radius=0, fg_color='transparent')

        self.rframe.grid(row=1, column=1, rowspan=4, columnspan=3, sticky="nsew")


        # Image Objects

        micPhoto = ctk.CTkImage(dark_image= Image.open("gui_icons\microphone_icon.png"), size=(80,80))

        camPhoto = ctk.CTkImage(dark_image= Image.open("gui_icons\camera_icon.png"), size=(80,80))

        playPhoto = ctk.CTkImage(dark_image=Image.open("gui_icons\play_icon.png"), size=(80,80))


        # MIC Button

        self.micButton = ctk.CTkButton(self.rframe, text="", image=micPhoto, command=self.mic, fg_color="transparent")

        self.micButton.grid(row=0, column=0, padx=(10, 10), pady=(10, 0))


        # CAM Button

        self.camButton = ctk.CTkButton(self.rframe, text="", image=camPhoto, command=self.cam, fg_color="transparent")


        # Browse Button

        self.browseButton = ctk.CTkButton(self.rframe, text="Browse", command=self.browse)

        self.browseButton.grid(row=1, column=4, padx=(10, 10), pady=(20, 0))#, pady=(20, 20))


        # File Path Entry

        self.fileEntry = ctk.CTkEntry(self.rframe, placeholder_text="File Path")

        self.fileEntry.grid(row=1, column=0, columnspan=2, padx=(10, 0), pady=(20, 0), sticky="ew")


        # # Clear Button

        # self.clearButton = ctk.CTkButton(self.rframe, text="Clear", command=self.clear)

        # # self.clearButton.grid(row=1, column=6)#, pady=(20, 20))


        # Covert Mode Button

        self.convButton = ctk.CTkButton(self.rframe, text="Convert", command=self.conv)     


        # Text Label

        self.textLabel = ctk.CTkLabel(master=self.rframe, text="Click on the Mic Button and Speak", font=ctk.CTkFont(size=16), anchor='e')#, weight="bold")

        self.textLabel.grid(row=0, column=1)#, padx=(20, 0), pady=(20, 0), sticky="nsew")


        # Translate Label

        # self.tranLabel = ctk.CTkLabel(master=self.rframe, text="", font=ctk.CTkFont(size=18, weight="bold"))
        

        # Inner Frame

        self.inframe = ctk.CTkScrollableFrame(master=self, fg_color='transparent')

        self.inframe.grid(row=2, column=1, rowspan=4, columnspan=3, padx=(20,20), pady=(20,20), sticky="nsew")


        # Images Canvas

        self.imgCanvas = ctk.CTkCanvas(master=self.inframe)

        # self.imgCanvas.grid(row=0, column=0, columnspan=4, sticky="nsew")


        # Play Button

        self.playButton = ctk.CTkButton(self.rframe, text="", image=playPhoto, command=self.play_audio, fg_color="white")



    # Defaults/Globals


    languages = ['Arabic', 'English', 'Hindi']
    

    # hi: Hindi

    codes = {'gtrans': ['ar', 'en', 'hi'],

    'gcloud': ['ar-EG', 'en-US', 'hi-IN']}


    guide = pd.read_csv('data\Arabic\Arabic_Letters_Guide.csv') # Loading the Arabic Letters Guide

    imgdir = 'data\Arabic\ArASL_Database_54K_Final\ArASL_Database_54K_Final'

    special_characters = ['@', '!', '?', '؟', '$', '%', '^', '*', '-', '_', ' ّ', ' ً', ' َ', ' ُ', ' ٌ', ' ِ', ' ٍ']

    ArSL = []

    inmode = "Microphone"

    srclang = "Arabic"

    destlang = "English"

    imgs = []

    label = ""

    string = ""

    pygame.mixer.init()  # initialise the pygame

    tran = False


    # Commands\Functions Section


    def inputType(self, input_mode: str):

        self.inmode = input_mode    

        if input_mode == "Microphone":

            self.camButton.grid_forget()
            # self.browseButton.grid_forget()
            # self.fileEntry.grid_forget()
            self.fileEntry.delete(0, ctk.END)
            # self.fileEntry.configure(placeholder_text="File Path")
            self.playButton.grid_forget()
            self.textLabel.configure(text="Click on the Mic Button and Speak")
            self.inframe.grid(row=2, column=1, rowspan=4, columnspan=3, padx=(20,20), pady=(20,20), sticky="nsew")
            self.micButton.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))
            # self.tranButton.configure(state='disabled')
            self.tStatusLabel.configure(text="Translator: OFF", text_color='red')
            self.convButton.grid_forget()
            # self.imgCanvas.delete("all")

        elif input_mode == "Camera":

            self.micButton.grid_forget()
            self.textLabel.configure(text="")
            # self.imgCanvas.grid_forget()
            self.inframe.grid_forget()
            # self.clearButton.grid_forget()
            # self.inframe.destroy()
            self.fileEntry.delete(0, ctk.END)
            # self.fileEntry.configure(placeholder_text="File Path")
            self.camButton.grid(row=0, column=0, padx=(20, 20), pady=(20, 20))
            self.convButton.grid_forget()
            # self.tranButton.configure(state='normal')

        else:

            self.camButton.grid_forget()
            self.browseButton.grid_forget()
            self.playButton.grid_forget()
            self.micButton.grid_forget()
            self.textLabel.configure(text="")
            self.fileEntry.delete(0, ctk.END)
            self.fileEntry.configure(placeholder_text="Enter Text")
            # self.tranButton.configure(state='normal')
            self.imgCanvas.grid_forget()
            self.inframe.grid_forget()
            self.inframe.grid(row=4, column=1, rowspan=4, columnspan=3, padx=(20,20), pady=(20,20), sticky="nsew")

            self.convButton.grid(row=1, column=4, padx=(10, 10), pady=(20, 0))

    # -------------------------------------------------
    
    def fromlang(self, language: str):
        self.srclang = language
    # ------------------------------------------------------

    def tolang(self, language: str):
        self.destlang = language
    # -----------------------------------------------------

    def mp3_text(self, file_audio):

        file_audio = sr.AudioFile(file_audio[0])
        # use the audio file as the audio source                                        
        r = sr.Recognizer()
        with file_audio as source:
            audio_text = r.record(source)

        # print(type(audio_text))
        text = r.recognize_google(audio_text, language='ar-EG')
        resh = reshape(text)
        rev = get_display(resh)
        self.textLabel.configure(text=rev, anchor='e')
        self.text_image(text)

    # ---------------------------------------------------------
    

    def tranMode(self):

        if self.switch_var.get() == 'ON':
            # self.tranLabel.grid(row=1, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
            self.tran=True
            self.tStatusLabel.configure(text="Translator: ON", text_color='green')
            self.srcLangLabel.grid(row=3, column=0, padx=(10, 0), pady=(10, 0), sticky="nsew")
            # row=2, column=0, padx=10, pady=(10, 0)
            self.srcLang_optionemenu.grid(row=4, column=0)#, padx=10, pady=(10, 0))
            self.srcLang_optionemenu.set("Arabic")  # Set Arabic as default
            self.destLangLabel.grid(row=5, column=0, padx=(10, 0), pady=(10, 0), sticky="nsew")
            self.destLang_optionemenu.grid(row=6, column=0)#, padx=10, pady=(10, 0))
            self.destLang_optionemenu.set("English")  # Set Arabic as default

            # self.inframe.grid_forget()
            self.playButton.grid_forget()
            self.imgCanvas.destroy()
            self.textLabel.configure(text="")
            self.fileEntry.delete(0, ctk.END)
            self.fileEntry.configure(placeholder_text="Enter Text")

            # t = self.fileEntry.get()
            self.convButton.configure(text='Translate', command=self.translate)


        else:

            self.tStatusLabel.configure(text="Translator: OFF", text_color='red')
            self.srcLangLabel.grid_forget()
            self.destLangLabel.grid_forget()
            self.srcLang_optionemenu.grid_forget()
            self.destLang_optionemenu.grid_forget()

            self.fileEntry.delete(0, ctk.END)
            self.fileEntry.configure(placeholder_text="Enter Text")
            self.convButton.configure(text='Convert', command=self.conv)

    #-------------------------------------------------


    def conv(self):

        txt = self.fileEntry.get()
        resh = reshape(txt)
        rev = get_display(resh)
        self.textLabel.configure(text=rev, anchor='e')
        self.text_speech(txt) # Arabic Text to Speech Conversion
        self.playButton.grid(row=1, column=5, padx=(10, 0), pady=(10, 0))
        self.imgCanvas.destroy()
        self.inframe.grid(row=2, column=1, rowspan=4, columnspan=3, sticky="nsew")#, padx=(10,10), pady=(10,0), sticky="nsew")
        self.text_image(txt) # Arabic Text to Images Conversion
    # ----------------------------------------------------------------------------

    def browse(self):  # Open Image Files

        tf = ctk.filedialog.askopenfilenames(
            initialdir=r".\data\\Test",
            title="Select",
            filetypes=(("All Files", "*.*"), ("Images", "*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG", ), ("Audios", "*.wav *.WAV"))
            )

        self.fileEntry.delete(0, ctk.END)
        self.fileEntry.insert(ctk.END, tf)
        
        self.imgCanvas.destroy()
        

        if self.inmode == "Microphone":
            self.mp3_text(tf)
        else:
            self.img_speech(list(tf))

    # -------------------------------------------------


    def modelpath(self, src):

        if src == "Arabic":
            return 'models\ArSLText.h5'
        elif src == "English":
            return 'models\ASLText.h5'
    # ------------------------------------------------


    def img_speech(self, fp): 

        #Loading the Model

        modelPath = self.modelpath(self.srclang)

        model = load_model(modelPath)
        

        for name in fp:

            img = cv2.imread(name)

            # self.imgs.append(img)

            resize = tf.image.resize(img, (256,256))

            np.expand_dims(resize, 0)

            yhat = model.predict(np.expand_dims(resize/255, 0))

            result = np.where(yhat[0] == np.amax(yhat[0]))


            letter = self.guide[self.guide["Index"] == result[0][0]]["Arabic_Letters"].iloc[0]

            self.string += letter


        # if self.tan:

        #     t = self.translate(self.string)

        #     self.tranLabel.configure(text=t)

        resh = reshape( self.string)

        rev = get_display(resh)
        

        self.textLabel.configure(text=rev)

        self.text_speech(self.string)  # Convert Text to Speech/Audio

        self.playButton.grid(row=3, column=0, padx=(20, 20), pady=(20,20))

    # -------------------------------------------------


    def image_list(self):

        for cla in os.listdir(self.imgdir):
            p = self.imgdir + '\\' + cla
            imgName = random.choice(os.listdir(p))
            imPath = p + '\\' + imgName
            image = cv2.imread(imPath)
            self.ArSL.append(image)
    # -------------------------------------------------

    def translate(self, txt=None):

        if self.inmode == "Text":
            txt = self.fileEntry.get()

        src_idx = self.languages.index(self.srclang)
        des_idx = self.languages.index(self.destlang)
        
        translator = Translator()
        translated = translator.translate(txt, src=self.codes['gtrans'][src_idx], dest=self.codes['gtrans'][des_idx])

        self.textLabel.configure(text="{} -> {}".format(translated.origin, translated.text))

        if self.inmode == "Text":
            self.text_speech(Text=translated.text, dest=self.codes['gtrans'][des_idx]) # Text to Speech Conversion
            self.playButton.grid(row=3, column=0, padx=(20, 20), pady=(20, 20))
            # self.imgCanvas.grid_forget()
            self.inframe.grid_forget()
            self.inframe.grid(row=4, column=1, rowspan=4, columnspan=3, padx=(20,20), pady=(20,20), sticky="nsew")
            self.text_image(translated.text) # Arabic Text to Images Conversion

        if self.inmode == "Microphone":
            return translated.text
    # -------------------------------------------------

    def mic(self): # Microphone Input
        src_idx = self.languages.index(self.srclang)
        # des_idx = self.languages.index(self.destlang)
        
        r = sr.Recognizer()
        mic = sr.Microphone()

        # Using Google Cloud API
        with mic as audio_file:
            r.adjust_for_ambient_noise(audio_file)
            audio = r.listen(audio_file)

            try:
                # language ar-EG, en-US
                text = r.recognize_google(audio, language=self.codes['gcloud'][src_idx])
                self.label = self.label + "\n" + text + " :لقد قلت"                

                if self.tran:#==True:

                    t = self.translate(txt=text)
                    self.text_image(t)
                    # break
                else:
                    self.text_image(text)
                    resh = reshape(text)
                    rev = get_display(resh)
                    self.textLabel.configure(text=rev, anchor='e')

                    # self.textLabel.configure(text=self.label)

            except Exception as e:
                self.label = self.label + "\n" + 'Error: ' + str(e)
            
            # resh = reshape(text)
            # rev = get_display(resh)
            # self.textLabel.configure(text=rev, anchor='e')

            # self.textLabel.configure(text=self.label)

    #--------------------------------------------------    

    def text_image(self, sentence):      

        if self.tran==True and self.destlang != "Arabic":
            return

        # Special Characters Removal
        for sp in self.special_characters:
            sentence = sentence.replace(sp, '')

        words = sentence.split(' ')
        encoded = []

        for word in words:
            list_code = []

            for letter in word:
                print(letter)
                # return
                # code = self.guide[self.guide['Arabic_Letters'] == letter]['Index'].values[0]#.iloc[0]
                # list_code.append(code)

            # encoded.append(list_code)   
        return
        self.image_list()  # Fetch an image for each letter 

        self.imgCanvas = ctk.CTkCanvas(master=self.inframe)
        self.imgCanvas.grid(row=0, column=0, columnspan=4, sticky="nsew")

        # Displaying the Images
        for l1 in encoded:
            self.fig = Figure(figsize=(15,3))#, frameon=False)
            l1.reverse()
            self.ax = self.fig.subplots(1, len(l1))

            for i in range(len(l1)):
                self.ax[i].imshow(self.ArSL[l1[i]], cmap='gray')
                self.ax[i].set_title(self.guide[self.guide['Index'] == l1[i]]['Arabic_Letters'].iloc[0])
                self.ax[i].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.imgCanvas)
            self.canvas.draw()
            self.canvas.get_tk_widget().grid()#sticky="nsew")

        # self.clearButton.grid(row=1, column=6)

    # -------------------------------------------------

    def cam(self):  # Camera Input

        source = cv2.VideoCapture(0)

        color_dict = (0,255,0)

        count = 0


        prev_val = 0

        letter = ""

        self.string = ""


        modelPath = self.modelpath(self.srclang)

        model = load_model(modelPath)


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
 

            yhat = model.predict(np.expand_dims(normalized, 0))

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


        resh = reshape(self.string)

        rev = get_display(resh)

        self.textLabel.configure(text=rev, anchor='e')


        # self.textLabel.configure(text=self.string)

        self.text_speech(self.string)  # Convert Text to Speech/Audio

        self.playButton.grid(row=3, column=0, padx=(20, 20), pady=(20, 20))

    # -------------------------------------------------
    

    def text_speech(self, Text, dest='ar'):

        # lang - ar, en


        txt_sound = gTTS(text=Text, lang=dest, slow=False)

        self.audioPath = 'audio/Audio_' + Text + '.mp3'

        txt_sound.save(self.audioPath)

        # self.playButton.grid(row=3, column=0, padx=(20, 20), pady=(20, 20))


    # -------------------------------------------------



    def play_audio(self):  # Using pygame


        pygame.mixer.music.load(self.audioPath)

        pygame.mixer.music.play(loops=0)


    # ----------------------------------------------------------------------



if __name__ == "__main__":


    app = App()
    app.mainloop()


