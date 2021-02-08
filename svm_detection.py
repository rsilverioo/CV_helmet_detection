import os
import cv2
import joblib
import numpy as np

from skimage import color
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression


# Function for sliding window method
def sliding_window(image, window_size, step_list):
    for y_cut in range(0, image.shape[0], step_list[1]):
        for x_cut in range(0, image.shape[1], step_list[0]):
            yield x_cut, y_cut, image[y_cut: y_cut + window_size[1], x_cut: x_cut + window_size[0]]


# Function to check if head rectangle is nearby person rectangle
def check_rectangles(a1, b1, a2, b2):

    # (x11, y11) (x12, y12) (x21, y21) (x22, y22)
    offset = round((y12-y11)/2)
    # Check x coordinates
    if a2[0] >= a1[0] and b2[0] <= b1[0]:
        # Check y coordinates
        if a2[1] >= a1[1]-offset and b2[1] <= b1[1]-offset:
            return True

    return False


# Image load
image_path = r"test/helmet_1.png"
orig = cv2.imread(image_path)
fix_scale = 500 / orig.shape[0]
orig = cv2.resize(orig, (0, 0), fx=fix_scale, fy=fix_scale, interpolation=cv2.INTER_AREA)
clone1 = orig.copy()
clone2 = orig.copy()
clone3 = orig.copy()
gray = color.rgb2gray(orig)

# Image Parameters
visualise = False
winSize1 = (64, 128)
winSize2 = (50, 50)
winStride = (2, 2)
downscale = 1.10

step_size = (5, 5)
persons_nms_threshold = 0.3
head_nms_threshold = 0.1

# Person model configuration
orientations1 = 8
pixels_per_cell1 = (16, 16)
cells_per_block1 = (2, 1)
block_norm1 = "L2"
feature_vector1 = True

# Head model configuration
orientations2 = 8
pixels_per_cell2 = (16, 16)
cells_per_block2 = (1, 1)
block_norm2 = "L2"
feature_vector2 = True

# List to store the detections
det_persons = []
det_heads = []

model_path = r"models"
person_model = joblib.load(os.path.join(model_path, 'person_final.model'))
head_model = joblib.load(os.path.join(model_path, 'head_final.model'))

#####################
# Check for persons #
#####################
scale = 0

for im_scaled in pyramid_gaussian(orig, downscale=downscale):

    if im_scaled.shape[0] < winSize1[1] or im_scaled.shape[1] < winSize1[0]:
        break

    for (x, y, window) in sliding_window(im_scaled, winSize1, step_size):
        if window.shape[0] != winSize1[1] or window.shape[1] != winSize1[0]:
            continue

        window = color.rgb2gray(window)
        fd = hog(window,
                 orientations=orientations1,
                 pixels_per_cell=pixels_per_cell1,
                 cells_per_block=cells_per_block1,
                 block_norm=block_norm1,
                 feature_vector=feature_vector1)
        fd = fd.reshape(1, -1)
        pred = person_model.predict(fd)

        if pred == 1:
            p_index = person_model.predict_proba(fd)[:, 1]
            if p_index > 0.65: # 0.65
                print([scale], x, y, p_index)
                det_persons.append((int(x * (downscale ** scale)), int(y * (downscale ** scale)), p_index,
                                    int(winSize1[0] * (downscale ** scale)), int(winSize1[1] * (downscale ** scale))))

        if visualise:
            visual = gray.copy()
            cv2.rectangle(visual, (x, y), (x + winSize1[0], y + winSize1[1]), (255, 0, 0), 2)
            cv2.imshow("Sliding window method", visual)
            cv2.waitKey(1)

    h, w = gray.shape
    gray = cv2.resize(gray, (int(w / downscale), int(h / downscale)), interpolation=cv2.INTER_AREA)
    scale += 1
    print(scale)

rect_persons = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in det_persons])
sc = [score[0] for (x, y, score, w, h) in det_persons]
print("sc: ", sc)
sc = np.array(sc)

pick1 = non_max_suppression(rect_persons, probs=sc, overlapThresh=persons_nms_threshold)

###################
# Check for heads #
###################
scale = 0

for im_scaled in pyramid_gaussian(orig, downscale=downscale):

    if im_scaled.shape[0] < winSize2[1] or im_scaled.shape[1] < winSize2[0]:
        break

    for (x, y, window) in sliding_window(im_scaled, winSize2, step_size):
        if window.shape[0] != winSize2[1] or window.shape[1] != winSize2[0]:
            continue

        window = color.rgb2gray(window)
        fd = hog(window,
                 orientations=orientations2,
                 pixels_per_cell=pixels_per_cell2,
                 cells_per_block=cells_per_block2,
                 block_norm=block_norm2,
                 feature_vector=feature_vector2)
        fd = fd.reshape(1, -1)
        pred = head_model.predict(fd)

        if pred == 1:
            p_index = head_model.predict_proba(fd)[:, 1]
            if p_index > 0.8: #0.8
                print([scale], x, y, p_index)
                det_heads.append((int(x * (downscale ** scale)), int(y * (downscale ** scale)), p_index,
                                  int(winSize2[0] * (downscale ** scale)), int(winSize2[1] * (downscale ** scale))))

        if visualise:
            visual = gray.copy()
            cv2.rectangle(visual, (x, y), (x + winSize2[0], y + winSize2[1]), (255, 0, 0), 2)
            cv2.imshow("Sliding window method", visual)
            cv2.waitKey(1)

    h, w = gray.shape
    gray = cv2.resize(gray, (int(w / downscale), int(h / downscale)), interpolation=cv2.INTER_AREA)
    scale += 1
    print(scale)

rect_heads = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in det_heads])
sc = [score[0] for (x, y, score, w, h) in det_heads]
print("sc: ", sc)
sc = np.array(sc)

pick2 = non_max_suppression(rect_heads, probs=sc, overlapThresh=head_nms_threshold)

###################
# Display results #
###################

for (x1, y1, x2, y2) in rect_persons:
    cv2.rectangle(clone1, (x1, y1), (x2, y2), (0, 255, 0), 2)
for (x1, y1, x2, y2) in rect_heads:
    cv2.rectangle(clone1, (x1, y1), (x2, y2), (255, 0, 0), 2)
cv2.imshow("Before NMS", clone1)
cv2.waitKey(0)

for (x1, y1, x2, y2) in pick1:
    cv2.rectangle(clone2, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(clone2, 'Person', (x1 - 2, y1 - 2), 1, 0.75, (0, 255, 0), 1)
for (x1, y1, x2, y2) in pick2:
    cv2.rectangle(clone2, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(clone2, 'Head', (x1 - 2, y1 - 2), 1, 0.75, (255, 0, 0), 1)
cv2.imshow('After NMS', clone2)
cv2.waitKey(0)

counter = 1
for (x11, y11, x12, y12) in pick1:
    for (x21, y21, x22, y22) in pick2:
        if check_rectangles((x11, y11), (x12, y12), (x21, y21), (x22, y22)):
            cv2.rectangle(clone3, (x11, y11), (x12, y12), (0, 255, 0), 2)
            cv2.putText(clone3, 'Person', (x11 - 2, y11 - 2), 1, 0.75, (0, 255, 0), 1)
            cv2.rectangle(clone3, (x21, y21), (x22, y22), (255, 0, 0), 2)
            cv2.putText(clone3, 'Head', (x21 - 2, y21 - 2), 1, 0.75, (255, 0, 0), 1)
            cv2.imwrite(os.path.join(r"crops", "person_" + str(counter) + ".png"), orig[y11:y12, x11:x12])
            cv2.imwrite(os.path.join(r"crops", "head_" + str(counter) + ".png"), orig[y21:y22, x21:x22])
            counter += 1
cv2.imshow('Rectangle overlap', clone3)
cv2.imwrite(r"final_detection.png", clone3)
cv2.waitKey(0)

cv2.destroyAllWindows()
