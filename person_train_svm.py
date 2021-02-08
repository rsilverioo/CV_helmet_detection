import os
import sys
import cv2
import glob
import joblib
import numpy as np
import random as rand
import sklearn.linear_model as svm

from skimage.feature import hog
from sklearn.utils import shuffle

# General configurations
MAX_HARD_NEGATIVES = 20000

# Model configuration
orientations = 8  # Ped = 9; Heads = 8
pixels_per_cell = (16, 16)  # Ped = 8; Heads = 6
cells_per_block = (2, 1)  # Ped = 3; Heads = 2
block_norm = "L2"
feature_vector = True

train_data = []
train_labels = []

# General dataset
pos_img_dir = r"images/pos_person"
neg_img_dir = r"images/train_64x128_H96/neg"
hnm_img_dir = r"images/train_64x128_H96/neg"


# Util functions
def sliding_window(image, win_size, step_size):
    for y in range(0, image.shape[0] - 128, step_size[1]):
        for x in range(0, image.shape[1] - 64, step_size[0]):
            yield x, y, image[y:y + win_size[1], x:x + win_size[0]]


def ten_random_windows(img_file):
    h, w = img_file.shape
    if h < 128 or w < 64:
        return []

    h = h - 128
    w = w - 64

    win_array = []

    for i in range(0, 10):
        x = rand.randint(0, w)
        y = rand.randint(0, h)
        win_array.append(img_file[y:y + 128, x:x + 64])

    return win_array


# Hard Negative Mining function
def hard_negative_mine(f_neg, win_size, win_stride):
    hard_negatives = []
    hard_negative_labels = []

    count = 0
    num = 0

    for imgfile in f_neg:
        hnm_img = cv2.imread(os.path.join(hnm_img_dir, imgfile), cv2.IMREAD_GRAYSCALE)

        for (x, y, im_window) in sliding_window(hnm_img, win_size, win_stride):
            features = hog(im_window,
                           orientations=orientations,
                           pixels_per_cell=pixels_per_cell,
                           cells_per_block=cells_per_block,
                           block_norm=block_norm,
                           feature_vector=feature_vector)
            if clf1.predict([features]) == 1:
                hard_negatives.append(features)
                hard_negative_labels.append(0)

                count = count + 1

            if count == MAX_HARD_NEGATIVES:
                return np.array(hard_negatives), np.array(hard_negative_labels)

        num = num + 1
        sys.stdout.write("\r" + "\tHard Negatives Mined: " + str(count) + "\tCompleted: " + str(
            round((count / float(MAX_HARD_NEGATIVES)) * 100, 4)) + " %")
        sys.stdout.flush()

    return np.array(hard_negatives), np.array(hard_negative_labels)


# Main Program
print("Reading Images")
pos_num_files = 0
neg_num_files = 0

# Load the positive features
for filename in glob.glob(os.path.join(pos_img_dir, "*.png")):
    fd = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    fd = cv2.resize(fd, (64, 128), interpolation=cv2.INTER_AREA)

    fd = hog(fd,
             orientations=orientations,
             pixels_per_cell=pixels_per_cell,
             cells_per_block=cells_per_block,
             block_norm=block_norm,
             feature_vector=feature_vector)
    train_data.append(fd)
    train_labels.append(1)
    pos_num_files += 1

print("Total Positive Images : " + str(pos_num_files))

# Load the negative features
for filename in glob.glob(os.path.join(neg_img_dir, "*.png")):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    windows = ten_random_windows(img)

    for win in windows:
        fd = hog(win,
                 orientations=orientations,
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block,
                 block_norm=block_norm,
                 feature_vector=feature_vector)
        train_data.append(fd)
        train_labels.append(0)
        neg_num_files += 1

print("Total Negative Images : " + str(neg_num_files))

train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_data, train_labels = shuffle(train_data, train_labels, random_state=0)  # Needed?

print("Images read and shuffled")
print("===========================")
print("Training Started")

clf1 = svm.LogisticRegression(C=0.5, dual=True, random_state=rand.randint(1, 100), solver='liblinear', max_iter=10000)
clf1.fit(train_data, train_labels)
joblib.dump(clf1, r'models/person_raw.model')

print("Training Completed")
print("===========================")
print("Hard Negative Mining")

# Configuration
window_stride = (8, 8)
window_size = (64, 128)

print("Maximum Hard Negatives to Mine: " + str(MAX_HARD_NEGATIVES))

neg_img_files = []
for (dirpath, dirnames, filenames) in os.walk(hnm_img_dir):
    neg_img_files.extend(filenames)
    break

hnm_data, hnm_labels = hard_negative_mine(neg_img_files, window_size, window_stride)
sys.stdout.write("\n")

hnm_data = np.concatenate((hnm_data, train_data), axis=0)
hnm_labels = np.concatenate((hnm_labels, train_labels), axis=0)

hnm_data, hnm_labels = shuffle(hnm_data, hnm_labels, random_state=0)

print("Final Samples dims: " + str(hnm_data.shape))
print("Retraining the classifier with final data")

clf2 = svm.LogisticRegression(C=0.5, dual=True, random_state=rand.randint(1, 100), solver='liblinear', max_iter=10000)
clf2.fit(hnm_data, hnm_labels)

print("Trained and dumping")
joblib.dump(clf2, r'models/person_final.model')
