package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/e-XpertSolutions/go-iforest/v2/iforest"
	randomforest "github.com/malaschitz/randomForest"
	"github.com/petar/GoMNIST"
)

func main() {

	//Get Training images
	images, labels, testimages, testlabels, prtestimages := GetMNIST()
	//Get anomaly scores and labels from isolation forest trained on whole dataset
	anomalyscores, anomalylabels := TrainDigitIForest(testimages, testlabels, false)

	//split into training and validation, roughly 80/20 split
	trainimages := images[0:50000][:]
	validationimages := images[50001:][:]
	trainlabels := labels[0:50000]
	validationlabels := labels[50001:]

	//Training of randomForest models on validation data
	forest1 := randomforest.Forest{}
	forest2 := randomforest.Forest{}
	forest3 := randomforest.Forest{}
	forest4 := randomforest.Forest{}
	forest1.Data = randomforest.ForestData{X: trainimages, Class: trainlabels}
	forest2.Data = randomforest.ForestData{X: trainimages, Class: trainlabels}
	forest3.Data = randomforest.ForestData{X: trainimages, Class: trainlabels}
	forest4.Data = randomforest.ForestData{X: trainimages, Class: trainlabels}
	forest1.Train(100)   //Model 1
	forest2.Train(1000)  //Model 2
	forest3.TrainX(100)  //Model 3
	forest4.TrainX(1000) //Model 4
	p1 := 0
	p2 := 0
	p3 := 0
	p4 := 0
	for i := 0; i < len(validationimages); i++ {
		vote1 := forest1.Vote(validationimages[i])
		vote2 := forest2.Vote(validationimages[i])
		vote3 := forest3.Vote(validationimages[i])
		vote4 := forest4.Vote(validationimages[i])
		bestV1 := 0.0
		bestV2 := 0.0
		bestV3 := 0.0
		bestV4 := 0.0
		bestI1 := -1
		bestI2 := -1
		bestI3 := -1
		bestI4 := -1
		for j, v := range vote1 {
			if v > bestV1 {
				bestV1 = v
				bestI1 = j
			}
		}
		for j, v := range vote2 {
			if v > bestV2 {
				bestV2 = v
				bestI2 = j
			}
		}
		for j, v := range vote3 {
			if v > bestV3 {
				bestV3 = v
				bestI3 = j
			}
		}
		for j, v := range vote4 {
			if v > bestV4 {
				bestV4 = v
				bestI4 = j
			}
		}
		if int(validationlabels[i]) == bestI1 {
			p1++
		}
		if int(validationlabels[i]) == bestI2 {
			p2++
		}
		if int(validationlabels[i]) == bestI3 {
			p3++
		}
		if int(validationlabels[i]) == bestI4 {
			p4++
		}

	}
	fmt.Printf("Model 1 Accuracy: %5.1f%%\n", 100.0*float64(p1)/float64(len(testimages))) //94.1%
	fmt.Printf("Model 2 Accuracy: %5.1f%%\n", 100.0*float64(p2)/float64(len(testimages))) //94.5%
	fmt.Printf("Model 3 Accuracy: %5.1f%%\n", 100.0*float64(p3)/float64(len(testimages))) //95.6%
	fmt.Printf("Model 4 Accuracy: %5.1f%%\n", 100.0*float64(p4)/float64(len(testimages))) //95.9%

	//Testing predictions on test dataset and compare with isolation forest
	os.Mkdir("imagesout", 0755)
	csvFile, _ := os.Create("./goScores.csv")
	writer := csv.NewWriter(csvFile)
	writer.Write([]string{"Index", "Label", "LabelPred", "Anomaly", "AnomalyScore"})
	predlabels := make([]int, 10000)
	p := 0
	n := 0
	sum_anomalyscores_correct := 0.0
	sum_anomalyscores_incorrect := 0.0
	for i := 0; i < len(testimages); i++ {
		vote := forest4.Vote(testimages[i]) //Performs prediction using best model (4)
		best := 0.0
		for j, v := range vote { //Finds predicted digit with highest confidence from random forest
			if v > best {
				best = v
				predlabels[i] = j
			}
		}
		if int(testlabels[i]) == predlabels[i] { //if prediction is correct
			p++
			sum_anomalyscores_correct = sum_anomalyscores_correct + anomalyscores[i]
		} else { //if prediction is incorrect, also print images
			n++
			sum_anomalyscores_incorrect = sum_anomalyscores_incorrect + anomalyscores[i]
			pic := image.NewGray(image.Rect(0, 0, 28, 28))
			pic.Pix = prtestimages[i]
			out, _ := os.Create("./imagesout/" + fmt.Sprintf("img%d_pred%d_true%d_anomaly%d", i, predlabels[i], testlabels[i], anomalylabels[i]) + ".png")
			png.Encode(out, pic)
		}
		writer.Write([]string{fmt.Sprintf("%d", i+1), fmt.Sprintf("%d", testlabels[i]), fmt.Sprintf("%d", predlabels[i]), fmt.Sprintf("%d", anomalylabels[i]), fmt.Sprintf("%f", anomalyscores[i])})
	}

	//Output test accuracy and performance report
	fmt.Printf("Test Accuracy: %5.1f%%\n", 100.0*float64(p)/float64(len(testimages)))                                   //95.5
	fmt.Printf("Average anomaly score for correctly classified images: %f\n", sum_anomalyscores_correct/float64(p))     //11.7
	fmt.Printf("Average anomaly score for incorrectly classified images: %f\n", sum_anomalyscores_incorrect/float64(n)) //11.8
	TestPerformanceReport(testlabels, predlabels)
}

func TestPerformanceReport(testlabels []int, predlabels []int) {
	//A lot of code built here using ChatGPT for confusion matrix and accuracy by digit
	matrix := make([][]int, 10)
	for i := range matrix {
		matrix[i] = make([]int, 10)
	}
	for i := 0; i < len(testlabels); i++ {
		actual := testlabels[i]
		predicted := predlabels[i]
		matrix[actual][predicted]++
	}
	fmt.Println("Confusion Matrix:")
	for i := 0; i < 10; i++ {
		fmt.Printf("  ")
		for j := 0; j < 10; j++ {
			fmt.Printf("%d\t", matrix[i][j])
		}
		fmt.Println()
	}
	numClasses := len(matrix)
	for class := 0; class < numClasses; class++ {
		truePositives := matrix[class][class]
		actualOccurrences := 0
		for i := 0; i < numClasses; i++ {
			actualOccurrences += matrix[class][i]
		}
		accuracy := float64(truePositives) / float64(actualOccurrences) * 100
		fmt.Printf("Digit %d Accuracy: %.2f%%\n", class, accuracy)
	}
}

func TrainDigitIForest(testimages [][]float64, testlabels []int, digit bool) (anomalyscores []float64, anomalylabels []int) {
	switch digit {
	case false: //Training of iforest on whole dataset
		anomalyscores = make([]float64, 10000)
		anomalylabels = make([]int, 10000)
		f := iforest.NewForest(1000, 256, 0.02) //initializes forest with 1000 trees, 256 samples, outlier ratio of 0.02
		f.Train(testimages)                     //uses BSTs to do partitioning
		f.Test(testimages)                      // calculates anomaly scores for each sample
		for i := 0; i < len(testlabels); i++ {
			anomalyscores[i] = f.AnomalyScores[i]
			anomalylabels[i] = f.Labels[i]
		}
	case true: //Training of iforest digit by digit - needs to be modified to keep indices from testimages
		anomalyscores = make([]float64, 10000)
		anomalylabels = make([]int, 10000)
		xlen := 0
		for digit := 0; digit < 10; digit++ {
			x, _ := DigitFilter(testimages, testlabels, digit) //creates new dataframe x for subset of training data for digit
			f := iforest.NewForest(100, 256, 0.01)             //reduced number of trees to 100 as fewer parameters than whole dataset
			f.Train(x)                                         //uses BSTs to do partitioning
			f.Test(x)                                          // calculates anomaly scores for each sample

			for i := 0; i < len(x); i++ {
				anomalyscores[i+xlen] = f.AnomalyScores[i]
				anomalylabels[i+xlen] = f.Labels[i]
			}
			xlen = xlen + len(x)

		}
	}
	return anomalyscores, anomalylabels
}

func GetMNIST() (images [][]float64, labels []int, testimages [][]float64, testlabels []int, prtestimages [][]uint8) {
	//Loads data into [][]float64 and [][]uint8 for printing
	train, test, _ := GoMNIST.Load("./data")
	images = make([][]float64, len(train.Images))
	labels = make([]int, len(train.Images))
	testimages = make([][]float64, len(test.Images))
	prtestimages = make([][]uint8, len(test.Images))
	testlabels = make([]int, len(test.Images))
	for i := 0; i < len(train.Images); i++ {
		images[i] = make([]float64, len(train.Images[0]))
		for p := range train.Images[0] {
			images[i][p] = float64(train.Images[i][p]) //converts integer pixel values to float64 for input to iforest
			labels[i] = int(train.Labels[i])
		}

	}
	for i := 0; i < len(test.Images); i++ {
		testimages[i] = make([]float64, len(test.Images[0]))
		prtestimages[i] = make([]uint8, len(test.Images[0]))
		for p := range test.Images[0] {
			testimages[i][p] = float64(test.Images[i][p]) //converts integer pixel values to float64 for input to iforest
			prtestimages[i][p] = uint8(test.Images[i][p]) //uint8 version needed for printing images
			testlabels[i] = int(test.Labels[i])
		}

	}
	return images, labels, testimages, testlabels, prtestimages
}

func DigitFilter(images [][]float64, labels []int, digit int) (x [][]float64, intx [][]uint8) {
	//For filtering out a dataframe of MNIST images by digit labels
	x = make([][]float64, 0)  //initialize output array
	intx = make([][]uint8, 0) //'intx' is the uint8 version for printing using image package
	for i := range images {
		if labels[i] == digit { //Calls out whether label at image index is equal to the input digit
			xx := make([]float64, len(images[0])) //temporary variable
			intxx := make([]uint8, len(images[0]))
			for j := 0; j < len(images[0]); j++ {
				xx[j] = float64(images[i][j])
				intxx[j] = uint8(images[i][j])
			}
			x = append(x, xx)
			intx = append(intx, intxx)
		}
	}
	return x, intx
}
