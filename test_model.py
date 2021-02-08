import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage import color
from sklearn import metrics
from skimage.feature import hog

# # # PERSONS # # #
# Image and model path
# test_type = 1
#
# pos_img_dir = r"images/test_64x128_H96/pos"
# neg_img_dir = r"images/test_64x128_H96/neg"
#
# clf = joblib.load(r'models/person_final.model')

# # # HEADS # # #
# Image and model path
test_type = 0

pos_img_dir = r"images/pos_head"
neg_img_dir = r"images/neg_head"

clf = joblib.load(r'models/head_final.model')

# Model configuration
orientations = 8
pixels_per_cell = (16, 16)
cells_per_block = (1, 1)  # Persons =(2,1); Heads = (1,1)
block_norm = "L2"
feature_vector = True

total_pos_samples = 0
total_neg_samples = 0


# Util functions
def crop_centre(img):
    h, w, d = img.shape
    cl = int((w - 64) / 2)
    t = int((h - 128) / 2)
    crop = img[t:t + 128, cl:cl + 64]
    return crop


def read_filenames():
    f_pos = []
    f_neg = []

    for (dirpath, dirnames, filenames) in os.walk(pos_img_dir):
        f_pos.extend(filenames)
        break

    for (dirpath, dirnames, filenames) in os.walk(neg_img_dir):
        f_neg.extend(filenames)
        break

    if test_type == 0:
        f_pos, f_neg = f_pos[round(len(f_pos)*0.7):len(f_pos)], f_neg[round(len(f_neg)*0.7):len(f_neg)]

    print("Positive Image Samples: " + str(len(f_pos)))
    print("Negative Image Samples: " + str(len(f_neg)))

    return f_pos, f_neg


def read_images(f_pos, f_neg):
    print("Reading Images")

    array_pos_features = []
    array_neg_features = []
    global total_pos_samples
    global total_neg_samples

    for imgfile in f_pos:
        img = cv2.imread(os.path.join(pos_img_dir, imgfile))
        if test_type:
            cropped = crop_centre(img)
            gray = color.rgb2gray(cropped)
        else:
            gray = color.rgb2gray(img)

        features = hog(gray,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm=block_norm,
                       feature_vector=feature_vector)
        array_pos_features.append(features.tolist())

        total_pos_samples += 1

    for imgfile in f_neg:
        img = cv2.imread(os.path.join(neg_img_dir, imgfile))
        if test_type:
            cropped = crop_centre(img)
            gray = color.rgb2gray(cropped)
        else:
            gray = color.rgb2gray(img)

        features = hog(gray,
                       orientations=orientations,
                       pixels_per_cell=pixels_per_cell,
                       cells_per_block=cells_per_block,
                       block_norm=block_norm,
                       feature_vector=feature_vector)
        array_neg_features.append(features.tolist())
        total_neg_samples += 1

    return array_pos_features, array_neg_features


################
# MAIN PROGRAM #
################
pos_img_files, neg_img_files = read_filenames()
pos_features, neg_features = read_images(pos_img_files, neg_img_files)

pos_result = clf.predict(pos_features)
pos_score = clf.predict_proba(pos_features)

neg_result = clf.predict(neg_features)
neg_score = clf.predict_proba(neg_features)

###################
# DISPLAY RESULTS #
###################
sample_test = np.concatenate((np.ones(total_pos_samples), np.zeros(total_neg_samples)))
sample_result = np.concatenate((pos_result, neg_result))
sample_score = np.concatenate((pos_score[:, 1], neg_score[:, 1]))
sample_range = len(sample_test)

# # # Classification Metrics # # #
print("\n ### Classification Metrics ###")

# Classification Report
report = metrics.classification_report(sample_test, sample_result)
print(report)

# Calculating false positives and negatives
true_positives = cv2.countNonZero(pos_result)
false_negatives = pos_result.shape[0] - true_positives

false_positives = cv2.countNonZero(neg_result)
true_negatives = neg_result.shape[0] - false_positives

print("True Positives: " + str(true_positives), "False Positives: " + str(false_positives))
print("True Negatives: " + str(true_negatives), "False Negatives: " + str(false_negatives))

# Logistic loss performance
log_results = metrics.log_loss(sample_test, sample_result)
print("Logloss: " + str(log_results))

# Graphic display of precision, recall y f1
accuracies = metrics.accuracy_score(sample_test, sample_result)
precisions = metrics.precision_score(sample_test, sample_result)
recalls = metrics.recall_score(sample_test, sample_result)

plt.figure(figsize=(12, 8))
axes = plt.gca()
axes.set_ylim([0, 1])

plt.scatter(np.arange(sample_range), sample_result, label='Accuracy score', color='blueviolet')
plt.plot(np.arange(sample_range), np.full(sample_range, np.mean(accuracies)), label='Mean accuracy score',
         linestyle='dotted', color='blueviolet')
plt.scatter(np.arange(sample_range), sample_result, label='Precision score', color='hotpink')
plt.plot(np.arange(sample_range), np.full(sample_range, np.mean(precisions)), label='Mean precision score',
         linestyle='dotted', color='hotpink')
plt.scatter(np.arange(sample_range), sample_result, label='Recall score', color='deepskyblue')
plt.plot(np.arange(sample_range), np.full(sample_range, np.mean(recalls)), label='Mean recall score',
         linestyle='dotted', color='deepskyblue')
plt.legend()
plt.show()

# Receiving Operating Characteristic Curve (ROC curve)
# Calculating ROC area-under-curve (ROC AUC) score
false_positive_rate, true_positive_rate, threshold = metrics.roc_curve(sample_test, sample_score)
print('ROC AUC score: ', metrics.roc_auc_score(sample_test, sample_score))

# Ploting ROC curve
plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - Linear SVM')
plt.plot(false_positive_rate, true_positive_rate)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# # # Regression Metrics # # #
print("\n ### Regression Metrics ###")
mae = metrics.mean_absolute_error(sample_test, sample_result)
mse = metrics.mean_squared_error(sample_test, sample_result)
r2s = metrics.r2_score(sample_test, sample_result)

print("Mean Absolute Error: " + str(mae))
print("Mean Squared Error: " + str(mse))
print("R Squared: " + str(r2s))



