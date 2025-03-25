import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

tfds.builder('bridge_orig', data_dir='/mnt/sh_flex_storage/home/yujiewan/embodied_ai/data/flex-new-out').download_and_prepare()

(ds_train), ds_info = tfds.load(
    'bridge_orig',
    split=['train'],
    shuffle_files=False,
    with_info=True
)

print("Dataset info:", ds_info)
print(type(ds_train))
# print(f"ds_train: {ds_train}")
ds_train_dataset = ds_train[0]

image_file = "/mnt/sh_flex_storage/home/yujiewan/embodied_ai/data/flex-test/pick-up-block/20250321_085908/images/20250321_090137.jpg"
image = Image.open(image_file)
test_img = np.array(image)
print(f"test_img: {test_img}")
print(f"image: {np.shape(test_img)}")
# exit(1)

for sample in ds_train_dataset.take(1):
    steps = sample['steps']
    print(f"steps: {steps}")
    for step in steps:
        image_tensor = step['observation']['image_0']
        image = tfds.as_numpy(image_tensor)

        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"image: {image}")

        plt.imshow(image)
        plt.axis('off')
        plt.show()

        break