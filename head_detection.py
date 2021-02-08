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


# Image load
image_path = r"test/helmet_2.jpg"
orig = cv2.imread(image_path)
fixscale = 500 / orig.shape[0]
orig = cv2.resize(orig, (0, 0), fx=fixscale, fy=fixscale, interpolation=cv2.INTER_AREA)
clone = orig.copy()
gray = color.rgb2gray(clone)

# Image Parameters
visualise = False
# win_size = (64, 128)
winSize = (50, 50)
winStride = (2, 2)
downscale = 1.10

step_size = (5, 5)
nms_threshold = 0.1

# Model configuration
orientations = 8  # Ped = 9; Heads = 11
pixels_per_cell = (16, 16)  # Ped = 8; Heads = 6
cells_per_block = (1, 1)  # Ped = 3; Heads = 2
block_norm = "L2"  # Ped = L2-Hys; Heads = L2-Hys
feature_vector = True  # Ped = True; Heads = False

# List to store the detections
detections = []

# The current scale of the image
model_path = r"models"
scale = 0
model = joblib.load(os.path.join(model_path, 'head_final.model'))

for im_scaled in pyramid_gaussian(orig, downscale=downscale):

    if im_scaled.shape[0] < winSize[1] or im_scaled.shape[1] < winSize[0]:
        break

    for (x, y, window) in sliding_window(im_scaled, winSize, step_size):
        if window.shape[0] != winSize[1] or window.shape[1] != winSize[0]:
            continue

        window = color.rgb2gray(window)
        fd = hog(window,
                 orientations=orientations,
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block,
                 block_norm=block_norm,
                 feature_vector=feature_vector)
        fd = fd.reshape(1, -1)
        pred = model.predict(fd)

        if pred == 1:
            p_index = model.predict_proba(fd)[:, 1]
            if p_index > 0.95:
                print([scale], x, y, p_index)
                detections.append((int(x * (downscale ** scale)), int(y * (downscale ** scale)), p_index,
                                   int(winSize[0] * (downscale ** scale)), int(winSize[1] * (downscale ** scale))))

        if visualise:
            visual = gray.copy()
            cv2.rectangle(visual, (x, y), (x + winSize[0], y + winSize[1]), (255, 0, 0), 2)
            cv2.imshow("Sliding window method", visual)
            cv2.waitKey(1)

    h, w = gray.shape
    gray = cv2.resize(gray, (int(w / downscale), int(h / downscale)), interpolation=cv2.INTER_AREA)
    scale += 1
    print(scale)

rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
print("sc: ", sc)
sc = np.array(sc)

pick = non_max_suppression(rects, probs=sc, overlapThresh=nms_threshold)
count = 0

for (x1, y1, x2, y2) in rects:
    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(orig, 'P:'+str(sc[count]), (x1 - 2, y1 - 2), 1, 0.75, (0, 0, 255), 1)
    count += 1
cv2.imshow("Before NMS", orig)
cv2.waitKey(0)

for (x1, y1, x2, y2) in pick:
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(clone, 'Head', (x1 - 2, y1 - 2), 1, 0.75, (0, 0, 255), 1)
cv2.imshow('After NMS', clone)
cv2.waitKey(0)

cv2.destroyAllWindows()
