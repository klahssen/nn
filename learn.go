package nn

import (
	"log"
	"math/rand"

	mat "github.com/klahssen/go-mat"
)

var dftLogger = &log.Logger{}

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
