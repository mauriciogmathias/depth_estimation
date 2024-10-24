import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import tensorflow as tf
import tensorflow_hub as hub

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

module = hub.load("https://www.kaggle.com/models/intel/midas/TensorFlow1/v2-1-small/1", tags=['serve'])

captured_frame = None

def capture_frame():
    global captured_frame
    ret, frame = camera.read()
    if ret:
        captured_frame = frame
        print("frame captured for depth prediction.")
        root.destroy()
        predict_depth(captured_frame)

def predict_depth(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0

    #resize the image to 256x256 for the model
    img_resized = tf.image.resize(img, [256, 256], method='bicubic', preserve_aspect_ratio=False)

    #change format from hwc to chw
    img_resized = tf.transpose(img_resized, [2, 0, 1])

    #convert to tensor and reshape
    img_input = img_resized.numpy().reshape(1, 3, 256, 256)
    tensor = tf.convert_to_tensor(img_input, dtype=tf.float32)

    #perform inference using the model
    output = module.signatures['serving_default'](tensor)

    #convert the output to a numpy array
    prediction = output['default'].numpy().reshape(256, 256)

    #resize the prediction back to the original image size
    prediction_resized = cv2.resize(prediction, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

    #normalize the depth map for visualization
    depth_min = prediction_resized.min()
    depth_max = prediction_resized.max()
    img_out = (255 * (prediction_resized - depth_min) / (depth_max - depth_min)).astype("uint8")

    #apply colormap to the depth map
    img_out_color = cv2.applyColorMap(img_out, cv2.COLORMAP_INFERNO)
    cv2.imwrite("depth_output.png", img_out_color)

    cv2.imshow('Depth Prediction', img_out_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def update_frame():
    ret, frame = camera.read()
    if ret:
        #convert the frame to rgb and then to imagetk format for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
    label.after(10, update_frame)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("can't open camera")
    exit()

desired_width = 640
desired_height = 480
camera.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

root = tk.Tk()
root.title("depth prediction from webcam")
root.geometry(f"{desired_width}x{desired_height+50}")

label = tk.Label(root)
label.pack()

capture_button = tk.Button(root, text="capture frame for depth prediction", command=capture_frame)
capture_button.pack(side=tk.BOTTOM)

update_frame()

root.mainloop()

camera.release()
cv2.destroyAllWindows()
