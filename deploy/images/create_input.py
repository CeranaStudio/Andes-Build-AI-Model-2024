import os
import argparse
import torch
from torchvision import transforms

def build_inputs(opts):
    from PIL import Image
    import numpy as np

    build_dir = os.path.abspath(opts.out_dir)

    # Download test image
    image_path = opts.input_image
    output_filename = os.path.basename(image_path).split(".")[0] + ".bin"
    image = Image.open(image_path).resize((224, 224))

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = transform_test(image)

    x = image.numpy()
    x = np.expand_dims(x, axis=0)

    print("x", x.shape)
    with open(os.path.join(build_dir, output_filename), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", required=True)
    parser.add_argument("-o", "--out_dir", default="images")
    opts = parser.parse_args()

    build_inputs(opts)