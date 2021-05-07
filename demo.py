import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import base64
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras import models
import tensorflow as tf
from PIL import Image

print("loading model...")
WasteNet = models.load_model("models/WasteNet.h5")


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

def predict(model, image):
    y_pred = model.predict(image)[0] * 100
    return y_pred

def load_image(filepath):
    image = Image.open(filepath)
    image = image.convert('RGB')
    image = image.resize((64,64))
    image = np.expand_dims(np.asarray(image) / 255, axis=0)
    image = tf.convert_to_tensor(image)
    return image

def format_image(filepath):
    image = Image.open(filepath)
    image = image.resize((256,256))
    image.save("temp/temp.png")

def plot_value_array(prediction_array):
    plt.grid(False)
    plt.xticks(range(6),["cardboard","glass","metal","paper","plastic","trash"])
    plt.ylabel("Confidence(%)")
    plt.xlabel("Catagory")
    thisplot = plt.bar(range(6), prediction_array, color="#777777")
    plt.ylim([0, 100])
    predicted_label = np.argmax(prediction_array)
    thisplot[predicted_label].set_color('blue')

#image = load_image("o_waste_images/glass/glass.jpg")
#prediction_array = predict(WasteNet,image)[0] * 100
#plot_value_array(prediction_array)
#plt.show()

# First the window layout in 2 columns
file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"
        )
    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
    [sg.Text("Predictions Plot")],
    [sg.Canvas(key="-CANVAS-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("WasteNet Demo", layout, finalize=True, element_justification="center", font="Helvetica 14",)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".png", ".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        filename = os.path.join(
            values["-FOLDER-"], values["-FILE LIST-"][0]
        )
        window["-TOUT-"].update(filename)
        format_image(filename)
        window["-IMAGE-"].update("temp/temp.png")
        if "figure_canvas_agg" in locals():
            figure_canvas_agg.get_tk_widget().forget()
            plt.close('all')
        imageData = load_image(filename)
        prediction_array = predict(WasteNet,imageData)
        plt.clf()
        plot_value_array(prediction_array)
        fig = plt.gcf()
        fig.set_size_inches(5,2)
        figure_canvas_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)

window.close()