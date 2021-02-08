import cv2
import numpy as np

# Image configuration
image_path = r'test/helmet_1.png'
image_scale = 500

# Canny configuration
canny_low_threshold = 50    # Recommended upper:lower ratio between 2:1 and 3:1
canny_high_threshold = 100  # Used to find initial segments of strong edges

# Hough configuration
dp = 2
minDist = 100   # Minimum distance between the centers of the detected c_list
param1 = 50     # default = 50 | High threshold
param2 = 18     # default = 18 | Accumulator threshold
minRadius = 35  # Minimum circle radius
maxRadius = 45  # Maximum circle radius

# HSV thresholds for color detection
fv_low_threshold = np.array([[0, 100, 180], [100, 100, 100], [15, 100, 120], [50, 50, 100],
                             [75, 135, 150], [0, 0, 130]])
fv_high_threshold = np.array([[20, 230, 255], [120, 255, 180], [50, 255, 255], [100, 255, 170],
                              [200, 232, 255], [200, 75, 250]])


# Util functions
def get_average(img_part, cx, cy, radius):
    maxy, maxx, channels = img.shape

    # Calculate the mask for the circle
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x * x + y * y
    mask = mask > radius * radius

    img_x = img_part.shape[1]
    img_y = img_part.shape[0]

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
    patch = img_part[max(0, cy - radius): min(img_y, cy + radius + 1),
                     max(0, cx - radius): min(img_x, cx + radius + 1), :]

    C = np.zeros(3)
    for c in range(channels):
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
    # Show the result
    cv2.imshow(img_name, img_cir)
    cv2.waitKey(0)


################
# MAIN PROGRAM #
################
if __name__ == '__main__':
    # Read the image
    img = cv2.imread(image_path)
    fix_scale = image_scale / img.shape[0]
    img = cv2.resize(img, (0, 0), fx=fix_scale, fy=fix_scale, interpolation=cv2.INTER_AREA)

    cv2.imshow('img_file', img)
    cv2.waitKey(0)

    img_g = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    img_g[:, :] = img[:, :, 0]

    # Canny edge detection
    edges = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    vis = cv2.Canny(img_g, canny_low_threshold, canny_high_threshold, 3)

    kernel = np.ones((3, 3), np.uint8)
    vis = cv2.morphologyEx(vis, cv2.MORPH_CLOSE, kernel)

    canny_result = np.copy(img_g)
    canny_result[edges.astype(np.bool)] = 0

    cv2.imshow("Canny image preview", vis)
    cv2.waitKey(0)

    # Hough transform
    circles = cv2.HoughCircles(vis, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2,
                               minRadius=minRadius, maxRadius=maxRadius)
    circles = circles[0, :, :]
    show_circles(img, 'Hough c_list preview', circles)

    # Get the average HSV color for each circle
    img_m = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    nbCircles = circles.shape[0]
    features = np.zeros((nbCircles, 3), dtype=np.int)
    for i in range(nbCircles):
        features[i, :] = get_average(img_m, int(circles[i, 0]), int(circles[i, 1]), int(circles[i, 2]))

    show_circles(img, 'HSV preview', circles, [str(features[i, :]) for i in range(nbCircles)])

    # Remove c_list based on the features detected
    selectedCircles = np.zeros(nbCircles, np.bool)
    for i in range(nbCircles):
        print(features[i, :])
        for j in range(fv_low_threshold.shape[0]):
            if all(fv_low_threshold[j, :] < features[i, :]) and all(fv_high_threshold[j, :] > features[i, :]):
                selectedCircles[i] = 1

    circles = circles[selectedCircles]
    show_circles(img, 'Final detection', circles)
