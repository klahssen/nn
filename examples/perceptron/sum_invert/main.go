package main

import (
	"fmt"
	"os"

	"github.com/klahssen/nn/examples/perceptron/generic"
	"github.com/klahssen/nn/internal/activation"
)

func target(x []float64) float64 {
	res := 0.0
	for i := range x {
		res += -1.0 * x[i]
	}
	return res
}
func main() {
	costFn := activation.Power(1.0, 2)
	inSize := 2
	learningRate := 0.1
	training := generic.GetRandomDataset(60, inSize, 1000, 1000, target)
	test := generic.GetRandomDataset(40, inSize, 1000, 20, target)
	p, err := generic.Learn(inSize, 2, learningRate, costFn, training, test)
	if err != nil {
		fmt.Printf("failed to learn target function: %s\n", err.Error())
	}
	fmt.Printf("\nTrained for sum of -x[i]\n\n")
	if err = p.JSON("config.json"); err != nil {
		fmt.Fprintf(os.Stderr, "failed to store json definition of the perceptron: %s", err.Error())
	}

}
