# This project was run on Google Collab from the dataset from Google drive 

# Collection of dataset from google drive
from google.colab import drive
drive.mount('/content/drive')

# Importation of essential Libraries 
import cv2
import numpy as np
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from IPython.display import Image, display

import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries



# || Cleaning the Images

# Removing of text on Image 
def remove_R_in_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 224, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    result = cv2.inpaint(image, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    return result

# Function to create the duplicate folder structure
def duplicate_structure(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for item in os.listdir(src_folder):
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        if os.path.isdir(src_path):
            duplicate_structure(src_path, dest_path)  

# Process images in the specified folder
def process_images(image_folder, output_folder):
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            result = remove_R_in_image(image_path)
            output_path = os.path.join(output_folder, 'inpainted_' + filename)
            cv2.imwrite(output_path, result)
            print(f'Processed {filename}')

# Main function to set up and organize the inpainting of the images
def main(input_folder):
    output_folder = 'processed_output' 
    duplicate_structure(input_folder, output_folder) 
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        output_subfolder = os.path.join(output_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for category in os.listdir(subfolder_path):
                category_path = os.path.join(subfolder_path, category)
                output_category_path = os.path.join(output_subfolder, category)

                if os.path.isdir(category_path):
                    process_images(category_path, output_category_path)

# Specify the input folder from Google Drive
input_folder = '/content/drive/MyDrive/projectDataset' 
main(input_folder)

# Copying the folder
src_folder = '/content/processed_output'  
dest_folder = '/content/drive/MyDrive/projectDataset/proceedoutput' 
shutil.copytree(src_folder, dest_folder)

# Definiing the directories 
base_dir = '/content/drive/MyDrive/projectDataset/proceedoutput'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')


# || Augemntating the Normal dataset 

augmented_dir = '/content/drive/MyDrive/projectDataset/proceedoutput/augemented'
if not os.path.exists(augmented_dir):
    os.makedirs(augmented_dir)

balance_augmentation = image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False,
    fill_mode='nearest'
)

normal_dir = os.path.join(train_dir, 'NORMAL')
image_normal_paths = [os.path.join(normal_dir, fname) for fname in os.listdir(normal_dir) if fname.endswith('.jpeg') or fname.endswith('.jpg')]

# Augment images
for image_path in image_normal_paths:
    img = tf.keras.preprocessing.image.load_img(image_path)
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    i = 0
    for batch in balance_augmentation.flow(x, batch_size=1):
        augmented_image = tf.keras.preprocessing.image.array_to_img(batch[0])
        save_path = os.path.join(augmented_dir, os.path.basename(image_path))
        augmented_image.save(save_path)
        break

augmented_files = os.listdir(augmented_dir)
print(f"Number of augmented images: {len(augmented_files)}")

# Transfer the augemented files to main dataset
augmented_files = os.listdir(augmented_dir)
for i, filename in enumerate(augmented_files):
    src = os.path.join(augmented_dir, filename)
    new_filename = f"aug_{i}_{filename}"
    dst = os.path.join(normal_dir, new_filename)

    try:
        shutil.move(src, dst)
        print(f"Moved {filename} to {normal_dir} as {new_filename}")
    except Exception as e:
        print(f"Error moving {filename}: {e}")

# Reinstate the directories 
base_dir = '/content/drive/MyDrive/projectDataset/proceedoutput'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

# || Preprocess the images 
batch_size = 16
img_height = 224
img_width = 224
labelFormat = 'binary'

# Load the train dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode= labelFormat,
    color_mode='grayscale'
)

# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode= labelFormat,
    color_mode='grayscale'
)

# Load test dataset 
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode= labelFormat,
    color_mode='grayscale'
)

# Normalize images
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = train_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)
test_ds = test_ds.map(normalize_img)

# Prefetch data for faster access
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# checking the class distribution 
def check_class_distribution(base_dir):
    for split in ['train', 'test', 'val']:
        split_dir = os.path.join(base_dir, split)
        print(f"Split: {split}")
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)
            if os.path.isdir(label_dir):
                num_images = len(os.listdir(label_dir))
                print(f"Class: {label}, Number of images: {num_images}")

check_class_distribution(base_dir)

# View images 
for images, labels in train_ds.take(1):
    numpy_images = images.numpy() 
    numpy_labels = labels.numpy()  
    batch_size = numpy_images.shape[0]
    plt.figure(figsize=(12, 12)) 
    for i in range(batch_size):
        plt.subplot(4, 4, i + 1)  
        plt.imshow(numpy_images[i].squeeze(), cmap='gray') 
        plt.title(f'Label: {numpy_labels[i]}')  
        plt.axis('off')
    plt.show()

# Compute class weights
all_labels = []
for _, labels in train_ds.unbatch():
    all_labels.append(labels.numpy())
all_labels = np.concatenate(all_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

print(f"Computed Class Weights: {class_weight_dict}")


# Building the Model: Custom CNN
# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
# Input layer for Custom CNN
input_layer = Input(shape=(224, 224, 1))

# Light Data Augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Model architecture
model = models.Sequential([
    input_layer,
    data_augmentation,  
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(64, activation='relu'),  
    layers.Dropout(0.3),  

    # Output Layer 
    layers.Dense(1, activation='sigmoid')  
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping for overfitting 
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Training the model
history = model.fit(train_ds, validation_data=val_ds, epochs=20, batch_size=16, class_weight=class_weight_dict, verbose=1, callbacks=[early_stopping, lr_scheduler])

# Evaluating the model 
# Plotting loss and accuracy
plt.figure(figsize=(12, 4))
# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Checking the test data
for images, labels in test_ds.take(1):
    preds = model.predict(images)

    # Plot images and predicted labels
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Pred: {int(preds[i] > 0.5)}, True: {labels[i].numpy()}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# checking test accuracy 
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Confusion Matrix
y_true = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
y_pred = (model.predict(test_ds) > 0.5).astype("int32").ravel()
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
# Checking the precision and all 
report = classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia'])
print(report)


# || ResNet Model
# For this preprocessing the colormode is not set to gray 
base_dir = '/content/drive/MyDrive/projectDataset/proceedoutput'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

batch_size = 16
img_height = 224
img_width = 224
labelFormat = 'binary'

# load train dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode= labelFormat #this is for the labels of puemonia and non puemonia

)
# Load validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode= labelFormat
)

# Load test dataset (using the smaller set as the test set)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode= labelFormat
)


# Normalize images
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = train_ds.map(normalize_img)
val_ds = val_ds.map(normalize_img)
test_ds = test_ds.map(normalize_img)

# Prefetch data for faster access
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Model Development: ResNet

# Load ResNet50 model without the top layer, using ImageNet weights
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Unfreeze some layers
for layer in base_model.layers[-5:]:
    layer.trainable = True

input_layer = tf.keras.Input(shape=(224, 224, 3))
x = base_model(input_layer)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=input_layer, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=30,
                    batch_size=16,
                    class_weight=class_weight_dict,
                    verbose=1,
                    callbacks=[early_stopping, lr_scheduler])

# Plotting loss and accuracy
plt.figure(figsize=(12, 4))

# Plot training & validation loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

# Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

for images, labels in test_ds.take(1):
    preds = model.predict(images)

    # Plot images and predicted labels
    plt.figure(figsize=(12, 12))
    for i in range(len(images)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Pred: {int(preds[i] > 0.5)}, True: {labels[i].numpy()}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# || Visualization: GRADCAM

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

last_conv_layer_name = "conv5_block3_out"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
img_size = (224, 224) 
img_path = "NORMAL2-IM-1440-0001.jpeg"
img_array = preprocess_input(get_img_array(img_path, size=img_size))

preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate Grad-CAM heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

def save_and_display_gradcam(img_path, heatmap, cam_path="new_image.jpg", alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)
    display(Image(cam_path))

save_and_display_gradcam(img_path, heatmap)


# || Visualization: UMAP

# Get Feature maps 
layer_names = [
    'conv1_conv',
    'conv2_block1_1_conv',
    'conv2_block1_2_conv',
    'conv2_block1_0_conv',
    'conv2_block1_3_conv',
    'conv2_block2_1_conv',
    'conv2_block2_2_conv',
    'conv2_block2_3_conv',
    'conv2_block3_1_conv',
    'conv2_block3_2_conv',
    'conv2_block3_3_conv'
]

layer_outputs = [model.get_layer(name).output for name in layer_names]
feature_model = Model(inputs=model.input, outputs=layer_outputs)

def extract_features(img_array, model):
    feature_maps = model.predict(img_array)
    return feature_maps

img_path = "person1597_bacteria_4189.jpeg"
img_array = preprocess_input(get_img_array(img_path, size=img_size))
feature_maps = extract_features(img_array, feature_model)
def flatten_feature_maps(feature_maps):
    flattened_maps = []
    for fmap in feature_maps:
        flattened_map = fmap.reshape(-1, fmap.shape[-1])
        flattened_maps.append(flattened_map)
    return flattened_maps

flattened_feature_maps = flatten_feature_maps(feature_maps)

# || Visualization: LIME

def predict_function(images):
    processed_images = preprocess_input(np.array(images))
    return model.predict(processed_images)
explainer = lime.lime_image.LimeImageExplainer()
img_path = "person1590_bacteria_4175.jpeg" 
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

explanation = explainer.explain_instance(img_array[0], predict_function, top_labels=5, num_samples=500)
first_labels = explanation.local_exp.keys()
label = list(first_labels)[0]
temp, mask = explanation.get_image_and_mask(label=label, positive_only=True, num_features=10, hide_rest=True)
plt.imshow(mark_boundaries(temp, mask))
plt.show()
