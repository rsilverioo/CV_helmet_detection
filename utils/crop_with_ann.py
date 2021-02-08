# Author: Ashar Fatmi
# Original source: https://github.com/AsharFatmi/Utility_functions_python

import os
import cv2
import json
from PIL import Image


def main():
    annotation_path = r'train/_annotations.coco.json'
    img_path = r'train'
    head_dir = r'head_train/'
    helmet_dir = r'helmet_train/'
    person_dir = r'person_train/'
    crap_dir = r'crap_train/'

    with open(annotation_path, 'r') as myfile:
        X = myfile.read()
    obj = json.loads(X)

    i = 1

    for annotation in obj['annotations']:
        image_id = annotation['image_id']

        for id in obj['images']:
            if id['id'] == image_id:
                img = id['file_name']
                imageObject = cv2.imread(os.path.join(img_path, img))

                x = annotation['bbox'][0]
                y = annotation['bbox'][1]
                width = annotation['bbox'][2]
                height = annotation['bbox'][3]

                if annotation['category_id'] == 1:  # Heads
                    # Square images of 50x50, discarding the ones that doesn't have the proper size
                    var = height - width
                    cropped = imageObject[y:y+height, x-var:x+height-var]

                    if cropped.shape[1] >= 50:
                        fix_scale = 50 / cropped.shape[0]
                        cropped = cv2.resize(cropped, (0, 0), fx=fix_scale, fy=fix_scale, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(head_dir, '{}_{}.png'.format(image_id, i)), cropped)

                elif annotation['category_id'] == 2:  # Helmets
                    # Square images of 50x50 pixels, discarding the ones that doesn't have the proper size
                    var = height - width
                    cropped = imageObject[y:y+height, x-var:x+height-var]

                    if cropped.shape[1] >= 50:
                        fix_scale = 50 / cropped.shape[0]
                        cropped = cv2.resize(cropped, (0, 0), fx=fix_scale, fy=fix_scale, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(helmet_dir, '{}_{}.png'.format(image_id, i)), cropped)

                elif annotation['category_id'] == 3:  # Persons
                    # Rectangular images of 64x128 pixels, discarding the ones that doesn't have the proper size
                    var = round(height/2 - width)
                    cropped = imageObject[y:y+height, x-var:x+round(height/2)-var]

                    if cropped.shape[1] >= 32:
                        fix_scale = 128 / cropped.shape[0]
                        cropped = cv2.resize(cropped, (0, 0), fx=fix_scale, fy=fix_scale, interpolation=cv2.INTER_AREA)

                        if cropped.shape[1] == 64:
                            cv2.imwrite(os.path.join(person_dir, '{}_{}.png'.format(image_id, i)), cropped)

                else:
                    cropped = imageObject.crop[y:y + height, x:x + width]

                    cv2.imwrite(os.path.join(crap_dir, '{}_{}.png'.format(image_id, i)), cropped)

                i += 1


if __name__ == "__main__":
    main()
