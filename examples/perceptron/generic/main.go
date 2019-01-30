package generic

import (
	"fmt"
	"math/rand"

	mat "github.com/klahssen/go-mat"
	"github.com/klahssen/nn"
	"github.com/klahssen/nn/internal/activation"
)

func sum(x []float64) float64 {
	res := 0.0
	for i := range x {
		res += x[i]
	}
	return res
}

type src struct {
	inSize int
	seed   int64
	max    int64
	n      int
	ind    int
	r      *rand.Rand
	fn     func(x []float64) float64
}

//Dataset represents anything able to provide Datapoints, one at a time
func (s *src) Next() *nn.Datapoint {
	if s == nil {
		panic("Dataset is nil")
	}
	if s.ind >= s.n {
		return nil
	}
	arr := make([]float64, s.inSize)
	for i := range arr {
		arr[i] = float64(s.r.Int63n(2*s.max)-s.max) / float64(s.max)
	}
	s.ind++
	return &nn.Datapoint{Inp: mat.NewM64(s.inSize, 1, arr), Exp: mat.NewM64(1, 1, []float64{s.fn(arr)})}
}
func (s *src) Size() int {
	if s == nil {
		panic("Dataset is nil")
	}
	return s.n
}
func (s *src) Left() int {
	return int(s.n - (s.ind + 1))
}
func (s *src) Reset() {
	s.r = rand.New(rand.NewSource(s.seed))
	s.ind = 0
}

//GetRandomDataset generate a dataset of random numbers between -1/+1. max is used to create the dispersion
func GetRandomDataset(seed int64, inSize int, max int64, nsamples int, fn func(x []float64) float64) nn.Dataset {
	if fn == nil {
		fn = sum
	}
	if nsamples < 0 {
		nsamples *= -1
	}
	if nsamples == 0 {
		nsamples = 10
	}
	s := &src{n: nsamples, max: max, inSize: inSize, seed: seed, fn: fn}
	s.Reset()
	return s
}

//Learn trains a new network that tries to learn target function
func Learn(inSize int, batchSize int, learningRate float64, costFn activation.F, training, test nn.Dataset) (*nn.Perceptron, error) {

	if inSize < 1 {
		return nil, fmt.Errorf("input size must be >=1")
	}
	if training == nil || training.Left() == 0 {
		return nil, fmt.Errorf("no training data")
	}
	if test == nil || test.Left() == 0 {
		return nil, fmt.Errorf("no test data")
	}
	if costFn.Func == nil {
		return nil, fmt.Errorf("cost function is nil")
	}
	if costFn.Deriv == nil {
		return nil, fmt.Errorf("cost derivative is nil")
	}
	fmt.Printf("create perceptron\n")
	ftype := "iden"
	fparams := []float64{}
	p, err := nn.NewPerceptron(inSize, learningRate, ftype, fparams, activation.Iden(), costFn)
	if err != nil {
		return nil, err
	}
	fmt.Printf("initialize weights and biases\n")
	p.UpdateCoefs([]float64{1.0, 1.0, 1.0})
	//fmt.Printf("test cost: cost(4)=%f\n", costFn.Func(4))
	//fmt.Printf("test deriv_cost: dcost(4)=%f\n", costFn.Deriv(4))
	//prepare training data
	nsamples := training.Left()

	//train
	e := 0.0
	avge := 0.0
	//iters := 10
	//batchSize := 2
	counter := 0
	//for iter := 0; iter < iters; iter++ {
	fmt.Printf("start training on %d samples ...\n", nsamples)
	if batchSize <= 0 {
		batchSize = nsamples
	}
	ok := true
	i := 0
	for ok {
		dp := training.Next()
		if dp == nil {
			ok = false
			continue
		}
		res, err := p.Compute(dp.Inp)
		if err != nil {
			panic(fmt.Errorf("training point[%d]: %s", i, err.Error()))
		}
		e = res - dp.Exp.At(0, 0)
		avge += e
		//fmt.Printf("input [%d]:\n x: %v, expected: %f, prediction: %v, erreur: %f\n", i, dp.Inp.GetData(), dp.Exp.GetData(), res, e)
		fmt.Printf("erreur: %.8f, cost: %.8f\n", e, costFn.Func(e))
		if (counter+1)%batchSize == 0 && counter > 0 {
			avge = avge / float64(batchSize)
			fmt.Printf("back propagation...\n")
			p.BackProp(dp.Inp, avge)
			avge = 0.0
		}
		counter++
		i++
	}
	nsamples = test.Left()
	fmt.Printf("\n\n\nevaluate on %d samples ...\n", nsamples)
	avge = 0.0
	ok = true
	i = 0
	for ok {
		dp := test.Next()
		if dp == nil {
			ok = false
			continue
		}
		res, err := p.Compute(dp.Inp)
		if err != nil {
			panic(fmt.Errorf("training point[%d]: %s", i, err.Error()))
		}
		e = res - dp.Exp.At(0, 0)
		avge += e
		//fmt.Printf("input [%d]:\n x: %v, expected: %f, prediction: %v, erreur: %f\n", i, dp.Inp.GetData(), dp.Exp.GetData(), res, e)
		fmt.Printf("erreur: %.8f, cost: %.8f\n", e, costFn.Func(e))
	}
	avge = avge / float64(nsamples)
	fmt.Printf("average error: %.8f\n", avge)
	return p, nil
}
