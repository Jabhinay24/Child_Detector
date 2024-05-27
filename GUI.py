import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2

# Load your pre-trained model (adjust the path as necessary)
model = tf.keras.models.load_model('Child_Detector(V2).h5')

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Child or Adult Detector")

        self.label = tk.Label(root, text="Load an image to detect if it's a child or an adult",font=("arial",15,"bold"))
        self.label.pack()

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.button = tk.Button(root, text="Load Image", font=("arial",10,"bold"),command=self.load_image)
        self.button.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.result_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = Image.open(file_path)
            img = img.resize((400, 400), Image.Resampling.LANCZOS)
            self.img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)

            # Preprocess the image and make prediction
            self.predict(file_path)

    def preprocess_image(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.resize(img, (64, 64))  # Resize to the input size of your model
        img = img / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def predict(self, file_path):
        img = self.preprocess_image(file_path)
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Assuming 0 is child and 1 is adult (modify as per your model)
        if predicted_class == 0:
            self.result_label.config(text="Prediction: Adult", font=("arial",20,"bold"),fg="blue")
        else:
            self.result_label.config(text="Prediction: Child", font=("arial",20,"bold"),fg="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
