

import cv2
import mediapipe as mp
from PIL import Image, ImageTk

from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
from predictions import words


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\priya\Downloads\ASL\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

import pyttsx3

def close():
   #win.destroy()
   window.destroy()

def read_text_aloud():
    # Initialize the pyttsx3 engine

    engine = pyttsx3.init()

    # Get the text from the entry widget
    text = words

    # Set the text to be spoken by the engine
    engine.say(text)

    # Run the engine
    engine.runAndWait()



window = Tk("PAB")

window.geometry("700x700")
window.configure(bg = "#000000")


canvas = Canvas(
    window,
    bg = "#000000",
    height = 700,
    width = 700,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_text(
    172.0,
    61.0,
    anchor="nw",
    text="ASL RECOGNITION",
    fill="#FBF9F9",
    font=("MontserratRoman Regular", 40 * -1)
)

image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    350.0,
    600.0,
    image=image_image_1
)

canvas.create_rectangle(
    89.0,
    190.0,
    610.0,
    194.0,
    fill="#F5F5F5",
    outline="")

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=read_text_aloud,
    relief="flat"
)
button_1.place(
    x=104.0,
    y=241.0,
    width=142.0,
    height=25.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=close,
    relief="flat"
)
button_2.place(
    x=466.0,
    y=241.0,
    width=142.0,
    height=25.0
)

# canvas.create_rectangle(
#     175.0,
#     289.0,
#     525.0,
#     489.0,
#     fill="#000000",
#     outline="#FBF9F9")

window.resizable(False, False)
window.mainloop()
