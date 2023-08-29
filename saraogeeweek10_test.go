package main

import (
	"testing"

	"github.com/e-XpertSolutions/go-iforest/v2/iforest"
	randomforest "github.com/malaschitz/randomForest"
)

func TestIsolationForest(t *testing.T) {

	//Data input
	images, labels, testimages, testlabels, _ := GetMNIST()
	if len(images) != 60000 {
		t.Errorf("Error in data input. Incorrect number of images")
	}
	if len(testimages[0]) != 784 {
		t.Errorf("There is an incorrect number of pixels per image")
	}
	if len(testimages) != 10000 {
		t.Errorf("Error in data input. Incorrect number of images")
	}
	if len(images[0]) != 784 {
		t.Errorf("There is an incorrect number of pixels per image")
	}

	//Training of iforest
	anomalyscores, anomalylabels := TrainDigitIForest(testimages, testlabels, false)
	if len(anomalyscores) != 10000 || len(anomalylabels) != 10000 {
		t.Errorf("Incorrect number of scores or labels calculated by iforest package")
	}
	for digit := 0; digit < 10; digit++ {
		x, _ := DigitFilter(images, labels, digit)
		if len(x) == 60000 {
			t.Errorf("Filtering did not occur")
		}
		if len(x) == 0 {
			t.Errorf("Filtering removed all images. Something wrong with DigitFilter")
		}
		//Training of forest
		f := iforest.NewForest(100, 256, 0.0001) //initializes forest with 1000 trees, 256 samples, dummy threshold value (not used but maybe classifying)
		f.Train(x)                               //uses BSTs to do partitioning
		f.Test(x)                                // calculates anomaly scores for each sample

		if len(f.AnomalyScores) != len(x) {
			t.Errorf("Incorrect number of scores calculated by iforest package for %d digit", digit)
		}
	}

	//Training of rforest
	forest := randomforest.Forest{}
	forest.Data = randomforest.ForestData{X: images, Class: labels}
	forest.Train(100)
	vote := forest.Vote(testimages[0])
	if len(vote) != 10 {
		t.Errorf("Incorrect number of classes in random forest output")
	}

}
