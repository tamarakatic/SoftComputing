from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_mldata
import glob

def train_knn():
    dataset = fetch_mldata('MNIST original')
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(dataset.data, dataset.target)
    return model

def sum_numbers(image_path, model):
    img = imread(image_path)
    gray = rgb2gray(img)
    binary = gray > 0

    # Label connected regions
    labelled = label(1 - binary)
    # Measure properties of labeled image regions
    properties = regionprops(labelled)

    regions = []
    for prop in properties:
        height = prop.bbox[2] - prop.bbox[0]
        width = prop.bbox[3] - prop.bbox[1]

        if (height == 28 and width == 28):
            x = prop.bbox[0]
            y = prop.bbox[1]
            temp = np.zeros((28, 28), np.float)
            temp[0:28, 0:28] = gray[x:x + 28, y:y + 28] * 255
            regions.append(temp)

    total = 0
    for reg in regions:
        flat = reg.astype('uint8').flatten().reshape(1, -1)
        total += model.predict(flat)

    return total

if __name__ == '__main__':
    with open('out.txt', 'w') as file:
        file.write('RA 7/2013 Tamara Katic\n')
        file.write('file\tsum\n')
        print '> Training KNN ...'
        model = train_knn()
        print '> Processing images ...\n'
        for image in glob.glob('images/*.png'):
            image = image.replace("\\","/")
            print '> ' + image
            file.write(image + '\t%.1f\n' % sum_numbers(image, model))
