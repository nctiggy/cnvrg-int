import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import requests
import os


exp_path = os.getenv('CNVRG_TASKS_PASSED_CONDITION')
model = load_model(f"{exp_path}/output/imagizer.model.h5")


def predict(file_url):
    print(file_url)
    r = requests.get(file_url)
    ext = r.headers['content-type'].split('/')[-1]
    with open(f"image.{ext}", 'wb') as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    img = image.load_img(f"image.{ext}", target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes[0][0])
    if classes[0][0] > 0.5:
        print("human")
        return "human"
    else:
        print("horse")
        return "horse"
