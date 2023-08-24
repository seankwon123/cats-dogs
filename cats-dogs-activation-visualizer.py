# %%
from tensorflow import keras



# %%
model = keras.models.load_model(
    "convnet_from_scratch_with_augmentation.x"
)

model.summary()

# %%
from tensorflow import keras
import numpy as np

# img_path = keras.utils.get_file(
#     fname="cat.jpg",
#     origin="https://img-datasets.s3.amazonaws.com/cat.jpg"
# )
img_path = './ivyVague.jpg'


def get_img_array(img_path, target_size):
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

img_tensor = get_img_array(img_path, target_size=(180, 180))

# %%
import matplotlib.pyplot as plt
plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

# %%
from keras import layers

layer_outputs=[]
layer_names=[]
for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_outputs.append(layer.output)
        layer_names.append(layer.name)
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# %%
first_layer_activation = activations[0]
print(first_layer_activation.shape)

# %%
plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis") 

# %%
model = keras.applications.xception.Xception(
    weights="imagenet",
    include_top=False
)

# for layer in model.layers:
#     if isinstance(layer, (keras.layers.Conv2D, keras.layers.SeparableConv2D)):
#         print(layer.name)

# %%
layer_name = "block3_sepconv1"
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

# %%
activation = feature_extractor(
    keras.applications.xception.preprocess_input(img_tensor)
)

# %%
import tensorflow as tf

def compute_loss(image, filter_index):
    activation = feature_extractor(image)
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

# %% [markdown]
# Gradient ascent step function using `GradientTape`

# %%
@tf.function
def gradient_ascent_step(image, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, filter_index)
    grads = tape.gradient(loss, image)
    grads = tf.math.l2_normalize(grads)
    image += learning_rate * grads
    return image

# %% [markdown]
# function to generate filter visualizations

# %%
img_width = 200
img_height = 200

def generate_filter_pattern(filter_index):
    iterations = 30
    learning_rate = 10.
    image = tf.random.uniform(
        minval=0.4,
        maxval=0.6,
        shape=(1, img_width, img_height, 3)
    )
    for i in range(iterations):
        image = gradient_ascent_step(image, filter_index, learning_rate)
    return image[0].numpy()

# %%
def deprocess_image(image):
    image -= image.mean()
    image /= image.std()
    image *= 64
    image += 128
    image = image[25:-25, 25:-25, :]
    return image

# %%
plt.axis("off")
plt.imshow(deprocess_image(generate_filter_pattern(filter_index=2)))

# %%



