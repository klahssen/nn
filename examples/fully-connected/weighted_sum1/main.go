package main

import (
	"fmt"
	"math/rand"
	"os"

	"github.com/klahssen/nn"
	"github.com/klahssen/nn/internal/activation"
	"github.com/klahssen/go-mat"
)

const (
	inSize = 3
	maxIter=1
	batchSize=1
)

func main() {
	fc, _ := nn.NewFC(inSize)
	configs := []*nn.LayerConfig{
		&nn.LayerConfig{Size: 1, FuncType: activation.FuncTypeIden},
	}
	if err := fc.SetLayers(configs...); err != nil {
		fmt.Fprintf(os.Stderr, "failed to set layers: %s\n", err.Error())
		os.Exit(1)
	}
	fct, err := nn.NewFCTrainer(fc, nil, nn.NewLr(0.1), maxIter, 0.001, activation.Power(0.5, 2))
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to construct new trainer: %s\n", err.Error())
		os.Exit(1)
	}
	training, validation, test := getDatasets()
	r, err := fct.TrainWithBackprop(rand.NewSource(42), 10, 0.5, batchSize, training, validation, test)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to train neural network: %s\n", err.Error())
		os.Exit(1)
	}
	//	fc.Info()
	tests := []struct {
		x []float64
		y []float64
	}{
		{[]float64{1.0, 2.0, 3.0}, target([]float64{1.0, 2.0, 3.0})},
		{[]float64{0.0, 2.0, 3.0}, target([]float64{0.0, 2.0, 3.0})},
		{[]float64{1.0, 2.0, 0.0}, target([]float64{1.0, 2.0, 0.0})},
	}
	for ind, t := range tests {
		pred, err := r.FeedForward(mat.NewM64(3,1,t.x))
		if err != nil {
			fmt.Printf("tests[%d]: failed to compute prediction: %s\n", ind, err.Error())
			continue
		}
		fmt.Printf("tests[%d]: input x=%v expected y=%v received %v\n", ind,t.x, t.y, pred.GetData())
	}
}

func target(x []float64) []float64 {
	res := x[0]*2.0 + 4*x[1] - 3*x[2]
	return []float64{res}
}

//returns training,validation and test datasets
func getDatasets() (nn.Dataset, nn.Dataset, nn.Dataset) {
	training := nn.NewRandomDataset(42, inSize, 1000, 500, target)
	test := nn.NewRandomDataset(50, inSize, 1000, 100, target)
	return training, nil, test
}
