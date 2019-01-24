package main

import (
	"fmt"
	"math/rand"

	mat "github.com/klahssen/go-mat"
	"github.com/klahssen/nn"
	"github.com/klahssen/nn/internal/activation"
)

func main() {
	net, err := nn.NewFFN(1)
	if err != nil {
		panic(err)
	}
	configs := []*nn.LayerConfig{
		&nn.LayerConfig{KeepState: false, Size: 1, FuncType: activation.FuncTypeIden, FuncParams: nil, F: activation.F{}},
	}
	net.SetLayers(configs...)
	net.SetLayerData(0, []float64{1, 1})
	if err = net.Validate(); err != nil {
		panic(err)
	}
	net.Info()
	//prepare training data
	nsamples := 10
	training := make([]*mat.M64, nsamples)
	r := rand.New(rand.NewSource(42))
	max := int64(50)
	rdm := 0.0
	for i := range training {
		rdm = float64(r.Int63n(2*max)-max) / float64(max)
		training[i] = mat.NewM64(1, 1, []float64{rdm})
	}

	//train
	for i, t := range training {
		res, err := net.Feed(t)
		if err != nil {
			panic(fmt.Errorf("training point[%d]: %s", i, err.Error()))
		}
		fmt.Printf("example [%d]: prediction: %v\n", i, res)
	}
}

/*
func (n *neuron) backprop(e float64) error {

}
*/
