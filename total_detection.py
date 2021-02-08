import os
import cv2
import joblib
import numpy as np

from skimage import color
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression

# Image load
image_path = r"test/helmet_1.png"
orig = cv2.imread(image_path)
fixscale = 500 / orig.shape[0]
orig = cv2.resize(orig, (0, 0), fx=fixscale, fy=fixscale, interpolation=cv2.INTER_AREA)
clone1 = orig.copy()
clone2 = orig.copy()
clone3 = orig.copy()
gray = color.rgb2gray(orig)

############################
# SVM Model Configurations #
############################

# Image Parameters
visualise = False
winSize1 = (64, 128)
winSize2 = (50, 50)
winStride = (2, 2)
downscale = 1.10

step_size = (5, 5)
persons_nms_threshold = 0.3
head_nms_threshold = 0.1

# Person HOG model configuration
orientations1 = 8
pixels_per_cell1 = (16, 16)
cells_per_block1 = (2, 1)
block_norm1 = "L2"
feature_vector1 = True

# Head HOG model configuration
orientations2 = 8
pixels_per_cell2 = (16, 16)
cells_per_block2 = (1, 1)
block_norm2 = "L2"
feature_vector2 = True

# List to store the detections
det_persons = []
det_heads = []

model_path = r"models_final"
person_model = joblib.load(os.path.join(model_path, 'person_final.model'))
head_model = joblib.load(os.path.join(model_path, 'head_final.model'))

###################################
# Helmet Detection Configurations #
###################################

# Canny configuration
canny_low_threshold = 50    # Recommended upper:lower ratio between 2:1 and 3:1
canny_high_threshold = 100  # Used to find initial segments of strong edges

# Hough configuration
dp = 2
minDist = 100   # Minimum distance between the centers of the detected c_list
param1 = 50     # High threshold
param2 = 18     # Accumulator threshold
minRadius = 35  # Minimum circle radius
maxRadius = 45  # Maximum circle radius

# HSV thresholds for color detection
fv_low_threshold = np.array([[0, 100, 180], [100, 100, 100], [15, 100, 120], [50, 50, 100],
                             [100, 150, 150], [0, 0, 130]])
fv_high_threshold = np.array([[20, 230, 255], [120, 255, 180], [50, 255, 255], [100, 255, 170],
                              [200, 232, 255], [200, 75, 250]])


##################
# Util functions #
##################
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


# Function to calculate the average color in the patch of image
def get_average(img, cx, cy, radius):
    maxy, maxx, channels = img.shape

    # Calculate the mask for the circle
    y_mask, x_mask = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x_mask * x_mask + y_mask * y_mask
    mask = mask > radius * radius

    img_x = img.shape[1]
    img_y = img.shape[0]

    if cx + radius >= img_x:
        xmax = cx + radius - img_x + 1
        mask = mask[:, :-xmax]
    if cx - radius < 0:
        xmin = radius - cx
        mask = mask[:, xmin:]
    if cy + radius >= img_y:
        ymax = cy + radius - img_y + 1
        mask = mask[:-ymax, :]
    if cy - radius < 0:
        ymin = radius - cy
        mask = mask[ymin:, :]

    # Truncating the circle in the image
    patch = img[max(0, cy - radius): min(img_y, cy + radius + 1), max(0, cx - radius): min(img_x, cx + radius + 1), :]

    C = np.zeros(3)
    for c in range(channels):
        # only False values are taken into account for the mean
        C[c] = np.mean(np.ma.masked_array(patch[:, :, c], mask))
    return C


def show_circles(img_cir, img_name, c_list, text=None):
    img_cir = np.copy(img_cir)
    nb_Circles = c_list.shape[0]

    for ind in range(nb_Circles):
        cv2.circle(img_cir, (int(c_list[ind, 0]), int(c_list[ind, 1])), int(c_list[ind, 2]), (255, 0, 0), 2, 8, 0)
    # Draw text
    if text is not None:
        for ind in range(nb_Circles):
            cv2.putText(img_cir, text[ind], (int(c_list[ind, 0]), int(c_list[ind, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255))

    cv2.imshow(img_name, img_cir)
    cv2.waitKey(0)


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
            if p_index > 0.65:
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

pick_persons = non_max_suppression(rect_persons, probs=sc, overlapThresh=persons_nms_threshold)

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
            if p_index > 0.8:
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

pick_heads = non_max_suppression(rect_heads, probs=sc, overlapThresh=head_nms_threshold)

####################
# Helmet detection #
####################
ind_helmet = np.array([])
cir_helmet = np.array([])
c_index = 0

for (x11, y11, x12, y12) in pick_persons:
    for (x21, y21, x22, y22) in pick_heads:
        if check_rectangles((x11, y11), (x12, y12), (x21, y21), (x22, y22)):
            ind_helmet = np.concatenate((ind_helmet, [0]), axis=0)

            img_person = orig[y21:y22, x21:x22]

            fix_scale = 100 / img_person.shape[0]
            img_person = cv2.resize(img_person, (0, 0), fx=fix_scale, fy=fix_scale, interpolation=cv2.INTER_AREA)

            # cv2.imshow('Cropped Person', img_person)
            # cv2.waitKey(0)

            img_g = np.zeros((img_person.shape[0], img_person.shape[1]), dtype=np.uint8)
            img_g[:, :] = img_person[:, :, 0]

            # Canny edge detection
            edges = np.zeros((img_person.shape[0], img_person.shape[1]), dtype=np.uint8)
            vis = cv2.Canny(img_g, canny_low_threshold, canny_high_threshold, 3)

            kernel = np.ones((3, 3), np.uint8)
            vis = cv2.morphologyEx(vis, cv2.MORPH_CLOSE, kernel)

            canny_result = np.copy(img_g)
            canny_result[edges.astype(np.bool)] = 0

            # cv2.imshow("Canny image preview", vis)
            # cv2.waitKey(0)

            # Circle Hough Transform
            circles = cv2.HoughCircles(vis, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2,
                                       minRadius=minRadius, maxRadius=maxRadius)
            if circles is None:
                ind_helmet[c_index] = 2
                c_index += 1
                continue

            circles = circles[0, :, :]
            # show_circles(img_person, 'Hough c_list preview', c_list)

            # Get the average HSV color for each circle
            img_m = cv2.cvtColor(img_person, cv2.COLOR_BGR2HSV)

            nbCircles = circles.shape[0]
            features = np.zeros((nbCircles, 3), dtype=np.int)
            for i in range(nbCircles):
                features[i, :] = get_average(img_m, int(circles[i, 0]), int(circles[i, 1]), int(circles[i, 2]))

            # show_circles(img_person, 'HSV preview', c_list, [str(features[i, :]) for i in range(nbCircles)])

            # Remove c_list based on the features detected
            selectedCircles = np.zeros(nbCircles, np.bool)

            for i in range(nbCircles):
                print(features[i, :])
                for j in range(fv_low_threshold.shape[0]):
                    if all(fv_low_threshold[j, :] < features[i, :]) and all(fv_high_threshold[j, :] > features[i, :]):
                        selectedCircles[i] = 1
                        ind_helmet[c_index] = 1

                if i == range(nbCircles) and ind_helmet[c_index] == 0:
                    ind_helmet[c_index] = 2

            circles = circles[selectedCircles]

            if cir_helmet.size == 0:
                cir_helmet = circles
            else:
                cir_helmet = np.concatenate((cir_helmet, circles), axis=0)

            # show_circles(img_person, 'Final detection', c_list)
            c_index += 1

###################
# Display results #
###################
c_index = 0

for (x11, y11, x12, y12) in pick_persons:
    for (x21, y21, x22, y22) in pick_heads:
        if check_rectangles((x11, y11), (x12, y12), (x21, y21), (x22, y22)):
            if ind_helmet[c_index] == 1:
                cv2.rectangle(clone2, (x11, y11), (x12, y12), (255, 0, 0), 2)
                cv2.putText(clone2, 'Person', (x11 - 2, y11 - 2), 1, 0.75, (255, 0, 0), 1)
                cv2.rectangle(clone2, (x21, y21), (x22, y22), (0, 255, 0), 2)
                cv2.putText(clone2, 'Helmet', (x21 - 2, y21 - 2), 1, 0.75, (0, 255, 0), 1)
            elif ind_helmet[c_index] == 0:
                cv2.rectangle(clone2, (x11, y11), (x12, y12), (255, 0, 0), 2)
                cv2.putText(clone2, 'Person', (x11 - 2, y11 - 2), 1, 0.75, (255, 0, 0), 1)
                cv2.rectangle(clone2, (x21, y21), (x22, y22), (0, 0, 255), 2)
                cv2.putText(clone2, 'Head', (x21 - 2, y21 - 2), 1, 0.75, (0, 0, 255), 1)

            c_index += 1

cv2.imshow('Final Result', clone2)
cv2.imwrite(r"final_result.png", clone2)
cv2.waitKey(0)

cv2.destroyAllWindows()
