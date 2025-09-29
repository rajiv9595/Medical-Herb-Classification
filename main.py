import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

try:
    RESAMPLE_MODE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_MODE = Image.LANCZOS

class_names = [
    'Alpinia Galanga (Rasna)', 'Amaranthus Viridis (Arive-Dantu)',
    'Artocarpus Heterophyllus (Jackfruit)', 'Azadirachta Indica (Neem)',
    'Basella Alba (Basale)', 'Brassica Juncea (Indian Mustard)',
    'Carissa Carandas (Karanda)', 'Citrus Limon (Lemon)',
    'Ficus Auriculata (Roxburgh fig)', 'Ficus Religiosa (Peepal Tree)',
    'Hibiscus Rosa-sinensis', 'Jasminum (Jasmine)',
    'Mangifera Indica (Mango)', 'Mentha (Mint)', 'Moringa Oleifera (Drumstick)',
    'Muntingia Calabura (Jamaica Cherry-Gasagase)', 'Murraya Koenigii (Curry)',
    'Nerium Oleander (Oleander)', 'Nyctanthes Arbor-tristis (Parijata)',
    'Ocimum Tenuiflorum (Tulsi)', 'Piper Betle (Betel)',
    'Plectranthus Amboinicus (Mexican Mint)', 'Pongamia Pinnata (Indian Beech)',
    'Psidium Guajava (Guava)', 'Punica Granatum (Pomegranate)',
    'Santalum Album (Sandalwood)', 'Syzygium Cumini (Jamun)',
    'Syzygium Jambos (Rose Apple)', 'Tabernaemontana Divaricata (Crape Jasmine)',
    'Trigonella Foenum-graecum (Fenugreek)'
]

model_filename = "C:/Users/RAJIV MEDAPATI/Documents/Medical_Herb_classification_project/resnet50_weights_th_dim_ordering_th_kernels_notop.h5"
loaded_model = None
try:
    loaded_model = tf.keras.models.load_model(model_filename)
except Exception:
    loaded_model = None

app = tk.Tk()
app.title("Medicinal Herb Classification")
app.geometry("800x600")
app.iconbitmap("app_icon.ico")

def insert_background():
    try:
        bg = Image.open("C:/Users/RAJIV MEDAPATI/Documents/Medical_Herb_classification_project/background_image.jpg")
        bg = ImageTk.PhotoImage(bg)
        lbl = tk.Label(app, image=bg)
        lbl.place(relwidth=1, relheight=1)
        lbl.image = bg
    except Exception:
        pass
insert_background()

result_label = tk.Label(app, text="Medical Herb Classification", font=("Helvetica", 16))
image_panel = tk.Label(app)

def train_model():
    global loaded_model
    train_dir = "C:/Users/RAJIV MEDAPATI/Documents/Medical_Herb_classification_project/medicalleafdatset"
    val_dir = train_dir
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    batch_size = 32
    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=batch_size, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224,224), batch_size=batch_size, class_mode='categorical')
    num_classes = train_gen.num_classes
    base = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(1024, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    loaded_model = Model(inputs=base.input, outputs=out)
    for layer in base.layers:
        layer.trainable = False
    loaded_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    loaded_model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples//batch_size,
        validation_data=val_gen,
        validation_steps=val_gen.samples//batch_size,
        epochs=5
    )
    loaded_model.save('medical_herb_detection.h5')
    messagebox.showinfo("Training Complete", "The model has been trained successfully.")

def save_model():
    global loaded_model
    if not loaded_model:
        messagebox.showwarning("Model Not Loaded", "Load or train a model before saving.")
        return
    fname = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 Files","*.h5")])
    if fname:
        try:
            loaded_model.save(fname)
            messagebox.showinfo("Model Saved", f"Saved to {fname}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def load_saved_model():
    global loaded_model
    fname = filedialog.askopenfilename(filetypes=[("H5 Files","*.h5")])
    if fname:
        try:
            loaded_model = tf.keras.models.load_model(fname)
            messagebox.showinfo("Model Loaded", f"Loaded {fname}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def classify_image():
    global loaded_model
    if not loaded_model:
        messagebox.showwarning("Model Not Loaded", "Load a model before making predictions.")
        return
    file_path = filedialog.askopenfilename(filetypes=[("Image files","*.jpg;*.jpeg;*.png;*.bmp"), ("All files","*.*")])
    if not file_path:
        return
    try:
        img = Image.open(file_path)
        img_resized = img.resize((224,224), RESAMPLE_MODE)
        photo = ImageTk.PhotoImage(img_resized)
        image_panel.config(image=photo)
        image_panel.image = photo
        arr = img_to_array(img_resized) / 255.0
        arr = np.expand_dims(arr, axis=0)
        preds = loaded_model.predict(arr)
        idx = np.argmax(preds)
        result_label.config(text=f"Leaf Name: {class_names[idx]}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

btn_frame = tk.Frame(app)
btn_frame.pack(pady=20)

tk.Button(btn_frame, text="Train Model", command=train_model).grid(row=0, column=0, padx=5)
tk.Button(btn_frame, text="Save Model", command=save_model).grid(row=0, column=50, padx=5)
tk.Button(btn_frame, text="Load Model", command=load_saved_model).grid(row=2, column=0, padx=5)
tk.Button(btn_frame, text="Classify Image", command=classify_image).grid(row=2, column=50, padx=5)

image_panel.pack(pady=10)
result_label.pack(pady=10)

tk.Label(app, text="By", font=("Helvetica", 20, "bold")).place(relx=0.5, rely=0.9, anchor=tk.CENTER)
tk.Label(app, text="M.RAJEEV REDDY", font=("Helvetica", 16, "bold")).place(relx=0.5, rely=0.95, anchor=tk.CENTER)

app.mainloop()
