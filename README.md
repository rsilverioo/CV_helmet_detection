# Computer vision algorithm for hardhat detection.

In this project, a computer vision algorithm for hardhat detection has been created by combining several techniques. The algorithm is divided into two parts, the first part being an identification of all persons in an image using SVM+HOG for the detection of persons and heads. And then a second detection of whether the detected person is wearing a helmet or not using Circle Hough Transform with Canny's algorithm to detect heads and finally colour detection in HSV space for helmet detection.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

The project has been programmed entirely in Python 3.8.5. The updated list of libraries can be seen in the *requirements.txt* file, but they are the following:

```
imutils==0.5.3
joblib==0.17.0
libsvm==3.23.0.4
matplotlib==3.2.0
numpy==1.19.3
opencv-python==4.4.0.46
Pillow==8.0.1
scikit-image==0.17.2
scikit-learn==0.23.2
scikit-optimize==0.8.1
scipy==1.5.4
```

Simply installing the above libraries using *pip* and downloading the repository will suffice, as it does not make use of any other components.

### Datasets

Two datasets have been used:

* INRIA Person Dataset [(link)](http://pascal.inrialpes.fr/data/human/)
* Hardhat dataset from Hardvard Dataverse [(link)](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/7CBGOS)

Mainly the INRIA images have been used for the training of persons and heads models, which have been cropped centred on the mentioned ROIs. However, the heads from the hardhats dataset have also been used to prevent the classifier from not being able to recognise the helmeted heads. The sizes used were 64x128 pixels for the human samples and 50x50 pixels for the head samples.

The hardhat dataset has been used mostly for the calibration of the second detection and for the final validation stage, as the objective is to detect who is wearing a helmet.

The programs take the images from the *images* folder, although you will have to check in the code if the paths are completely correct.

### Usage example

The project is divided into different programs with self-explanatory names, but here is a list of clarifications:

* The training of SVM models is done with files named *person/helmet_train_svm.py*, which stores the final models in the *models* folder. A quick verification of the model can be done using the *person/helmet_detection.py* files to check that the model works correctly.
* When you have both SVM models (people and heads) you can test it using *svm_detection.py* and check their accuracy with the program *test_model.py*.
* For helmet detection you can use the program *helmet_detection.py*, which will show the steps one by one in order to calibrate all the settings.
* Finally, the result of the complete detection can be obtained using the *total_detection.py* file, which will make use of part of the previous programs.

In the *utils* folder there are two small programs to obtain the minimum and maximum values of the HSV space to perform the colour detection, and another one to crop images from a dataset using its COCO annotations.

In the *test* folder there are some images in case you want to test the algorithms.

To summarise, to train the algorithm from scratch and perform a complete detection, the following steps must be followed:

1. Train the two SVM models with *person_train_svm.py* and *helmet_train_svm.py*.
2. (Avoidable) Check if models work and are valid with *test_model.py*.
3. (Avoidable) Calibrate and check helmet detection settings with *helmet_detection.py*.
4. Get the final detection with *total_detection.py*.

## Built With

* [scikit-learn](https://scikit-learn.org) - Machine Learning in Python
* [scikit-image](https://scikit-image.org) - Image processing in Python
* [OpenCV](https://opencv.org) - Open source computer vision and machine learning software library

## Contributing

1. Fork it (<https://github.com/rsilverioo/CV_helmet_detection/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## Authors

* **Rodrigo Silverio** - *Full work* - [rsilverioo](https://github.com/rsilverioo)

See also the list of [contributors](https://github.com/rsilverioo/CV_helmet_detection/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Thanks to the CV examples of other colleagues
* Thanks to Adrian Rosebrock for his **awesome** page and resources [(link)](https://www.pyimagesearch.com)
