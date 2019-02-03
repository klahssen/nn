package nn

import (
	"log"
	"math/rand"
	"os"

	mat "github.com/klahssen/go-mat"
)

var dftLogger = log.New(os.Stderr, "", log.LstdFlags)

//Logger interface, injected in the
type Logger interface {
	Printf(format string, v ...interface{})
}

//Datapoint holds input data and expected output
type Datapoint struct {
	Inp *mat.M64 //[]float64
	Exp *mat.M64 //[]float64
}

//Dataset represents anything able to provide Datapoints, one at a time
type Dataset interface {
	Next() *Datapoint
	Size() int
	Left() int
	Reset()
}

//RandomDataset returns random values from the seed
type RandomDataset struct {
	inSize int
	seed   int64
	max    int64
	n      int
	ind    int
	r      *rand.Rand
	fn     func(x []float64) []float64
}

//Next to implement Dataset interface
func (s *RandomDataset) Next() *Datapoint {
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
	return &Datapoint{Inp: mat.NewM64(s.inSize, 1, arr), Exp: mat.NewM64(1, 1, s.fn(arr))}
}

//Size to implement Dataset interface
func (s *RandomDataset) Size() int {
	if s == nil {
		panic("Dataset is nil")
	}
	return s.n
}

//Left to implement Dataset interface
func (s *RandomDataset) Left() int {
	return int(s.n - (s.ind + 1))
}

//Reset to implement Dataset interface
func (s *RandomDataset) Reset() {
	s.r = rand.New(rand.NewSource(s.seed))
	s.ind = 0
}

func sum(x []float64) []float64 {
	res := 0.0
	for i := range x {
		res += x[i]
	}
	return []float64{res}
}

//NewRandomDataset generates a dataset of random numbers between -1/+1. max is used to create the dispersion
func NewRandomDataset(seed int64, inSize int, max int64, nsamples int, fn func(x []float64) []float64) Dataset {
	if fn == nil {
		fn = sum
	}
	if nsamples < 0 {
		nsamples *= -1
	}
	if nsamples == 0 {
		nsamples = 10
	}
	s := &RandomDataset{n: nsamples, max: max, inSize: inSize, seed: seed, fn: fn}
	s.Reset()
	return s
}

//selectDrops provides a random selection of neurons to be deactivated
func selectDrops(r rand.Source, dropSize, fleetSize int) map[int]struct{} {
	if fleetSize <= 0 {
		return nil
	}
	if dropSize < 0 {
		dropSize *= -1
	}
	if dropSize > fleetSize {
		dropSize = fleetSize
	}
	if dropSize == 0 {
		return nil
	}
	m := map[int]struct{}{}
	counter := 0
	pick := 0
	for counter < dropSize {
		pick = int(r.Int63() % int64(fleetSize))
		if _, ok := m[pick]; !ok {
			m[pick] = struct{}{}
			counter++
		}
	}
	return m
}
