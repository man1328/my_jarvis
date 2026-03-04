import torch
import torch.nn.functional as F
from cnn_model import CNN
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

# 1. Load the Trained Brain 🧠
model = CNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  # Set to evaluation mode

class App:
    def __init__(self, root):
        self.classes = [
            "0", "1", "2", "3", "4",
            "5", "6", "7", "8", "9"
        ]

        self.root = root
        self.root.title("Digit Draw 🎨")

        # A canvas to draw on (white background)
        self.canvas = tk.Canvas(root, width=300, height=300, bg='white')
        self.canvas.pack()

        # Button to clear the screen
        self.btn_clear = tk.Button(root, text="Clear", command=self.clear)
        self.btn_clear.pack()

        # Label to show prediction
        self.label_pred = tk.Label(root, text="Draw a digit!", font=("Helvetica", 24))
        self.label_pred.pack()

        # Create a blank image to draw on in memory (same size as canvas)
        self.image = Image.new("L", (300, 300), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.predict)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle((0, 0, 300, 300), fill=255)
        self.label_pred.config(text="Draw a digit!")

    def paint(self, event):
        # Draw on the GUI canvas
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=20)

        # Draw on the hidden memory image (to be sent to AI)
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def predict(self, event):
        classes = ["0", "1", "2", "3", "4",
                   "5", "6", "7", "8", "9"]

        # 1. Process the image
        img_inverted = ImageOps.invert(self.image)
        bbox = img_inverted.getbbox()

        if bbox:
            # Crop to the drawing
            img_cropped = img_inverted.crop(bbox)

            # Resize to 20x20 and pad to 28x28
            img_resized = img_cropped.resize((20, 20), resample=Image.Resampling.LANCZOS)
            final_img = Image.new("L", (28, 28), 0)
            final_img.paste(img_resized, (4, 4))

            # 2. Convert to Tensor (THE MISSING LINK!)
            img_array = np.array(final_img)
            img_tensor = torch.tensor(img_array, dtype=torch.float32)
            img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
            img_tensor = img_tensor.view(1, 1, 28, 28)

            # 3. Predict
            with torch.no_grad():
                # Setting model to eval mode ensures dropout is OFF
                model.eval()
                output = model(img_tensor)
                probabilities = torch.exp(output)[0]

                # Get the Top 3 predictions
                top_probs, top_indices = torch.topk(probabilities, 3)

            # 4. Display Results
            result_text = "Analysis:\n"
            for i in range(3):
                name = classes[top_indices[i]]
                conf = top_probs[i].item()
                result_text += f"{i + 1}. {name} ({conf:.1%})\n"

            self.label_pred.config(text=result_text, justify="left")
        else:
            self.label_pred.config(text="Draw something first!")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
