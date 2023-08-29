# Image Classification towards an OCR pipeline

## Project Summary
This project aims to build an image processing demonstraion using Go looking to implement an OCR pipeline from image capture to recognition using classification. For this demonstration, classification is performed using random forests([randomForest](https://github.com/malaschitz/randomForest)) employed on The Modified National Institute of Standards and Technology (MNIST) dataset using the [GoMNIST driver](https://github.com/kuroko1t/GoMNIST). MNIST comprises of 60 thousand training observations and 10 thousand test observations of handwritten digits. Our use of MNIST utilized a 80/20 split in the training set for validation and all of the test observations for testing. Four different random forest models were evaluated using the validation dataset by varying the number of trees and partitioning scheme. Results are compared using an isolation forest ([go-iforest](https://github.com/e-XpertSolutions/go-iforest)) trained on all of the test observations. Misclassified images from the test set are [printed](./imagesout) for further evaluation using Go's [image package](https://pkg.go.dev/image). The best-performing model utilized 1000 trees and had **96% accuracy on the hold-out test dataset** and comparable accuracy for each digit. The average anomaly score was expectedly higher for the misclassified images.

Like in other projects, we see that Go is extremely fast in classifying and recognizing the images. The MNIST dataset is ideal for benchmarking an OCR pipeline. Development would focus on integrating new observations into the random forest model by using the [AddDataRow function](https://github.com/malaschitz/randomForest/blob/82dce2f56816/forest.go#L119). A key next step would be to integrate real-time image recognition for robotic process automation. This will be accomplished using [GoCV](https://gocv.io/), which leverages OpenCV, a well known computer vision library built by Intel in 1999. 

Other deep learning and image processing applications were considered but ultimately dropped due to various reasons outlined below:
-[Gorgonia](https://gorgonia.org/tutorials/mnist/): Gorgonia is considered the gold standard and has support for convolutional neural networks. However, using tensors for modeling was not straightforward.
-[go-deep](https://github.com/patrikeh/go-deep): Active repository for deep neural network. This was our initially preferred modelling method but ran into unexpected behavior. The validation accuracy was >95% but the test accuracy was ~11% indicating overfitting. More work is needed on this package.
-[golearn](https://github.com/sjwhitworth/golearn): Pacakge includes various machine learning models for easy comparison. However data input requires using FixedDataGrid format or csv files.
-[pigo](https://github.com/esimov/pigo): This package is a great alternative to OpenCV for image processing. However, it requires additional development to connect with a camera which is why we propose a system using GoCV instead.


## Results

Model 1 Results:  94.3%
Model 2 Results:  94.5%
Model 3 Results:  95.7%
Model 4 Results:  95.9%
**Model 4 performs the best with 1000 trees and "extra"-random partitioning**

Test Accuracy:  95.5%
Average anomaly score for correctly classified images:  11.7
Average anomaly score for incorrectly classified images:  11.8
**Great accuracy on the testing dataset. Slightly higher anomaly score for incorrectly classified images**

Confusion Matrix:
  968   0       0       0       0       3       5       1       3       0
  0     1121    3       4       0       1       3       1       2       0
  8     0       977     6       9       3       5       13      9       2
  3     0       14      955     1       17      0       9       7       4
  2     0       2       0       931     0       8       2       5       32
  7     2       0       20      1       836     10      1       11      4
  11    3       1       0       6       4       929     0       4       0
  2     6       27      2       1       0       0       971     4       15
  5     0       5       6       7       8       5       4       917     17
  5     4       1       12      11      7       4       6       11      948

Class 0 Accuracy: 98.78%
Class 1 Accuracy: 98.77%
Class 2 Accuracy: 94.67%
Class 3 Accuracy: 94.55%
Class 4 Accuracy: 94.81%
Class 5 Accuracy: 93.72%
Class 6 Accuracy: 96.97%
Class 7 Accuracy: 94.46%
Class 8 Accuracy: 94.15%
Class 9 Accuracy: 93.95%
**Digits 0, 1 and 6 have higher accuracy than other digits.**


## Under the go directory

**saraogeeweek10.go:** Main routine to load MNIST dataset, train random forests and compare tests with isolation forests.

**saraogeeweek10_test.go:** does unit tests for loading dataset and prediction dimensions.

**Week10** executable for saraogeeweek10.go on whole dataset cross-compiled Go code for Mac/Windows. 

## Installation and Running

Download or git clone this project onto local machine into folder on local machine.
```
git clone https://github.com/asaraog/msds431week10.git

cd msds431week10
./Week10
cd imagesout
ls
```
Images are printed in a new directory 'imagesout' with the name coressponding to imageID, predicted digit, true digit and a boolean score for whether or not it is classified as anomalous. A csv file titiled ('goScores.csv')[./goScores.csv] is also created with information for all of the images and an additional column for the anomaly score.

## Our Go expertise

Our other projects highlight our expertise in using Go for various machine learning tasks with comparisons to Python/R:

[Stats in Go](https://github.com/asaraog/msds431week2): Runtimes were compared for linear regression in the [Anscombe Quartet](https://www.sjsu.edu/faculty/gerstman/StatPrimer/anscombe1973.pdf) with Python/R.
[Websites using Go](https://github.com/asaraog/msds431week3): A promotional website is created for a company titled []'AutoNotes'](https://autonotes.netlify.app/) using Hugo.
[Command line applications in Go](https://github.com/asaraog/msds431week4): Summary statistics for the California Housing Prices study are compared with Python/Go.
[Scraping the web using Go](https://github.com/asaraog/msds431week5): Scrapes the Web for Wikipedia pages and compare results with Python's scrapy package.
[Linear Regression using Go](https://github.com/asaraog/msds431week6): Runtimes were compared with and without concurrency for linear regression in the Boston Housing Study.
[Isolation Forests](https://github.com/asaraog/msds431week7): Isolation forests are trained on the MNIST dataset and compared with R/Python on the whole dataset and by digit for correlation and runtime.
[Desktop Applications](https://github.com/asaraog/msds431week8): A protype desktop application is created using Wails/Svelte and Vale for assisted writing.
[Natural Language Processing](https://github.com/asaraog/msds431week9): A protype desktop application is created using Wails/Svelte for a simple lookup function from a corpus.






