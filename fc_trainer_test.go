package nn

import (
	"fmt"
	"log"
	"testing"

	"github.com/klahssen/nn/internal/activation"
	"github.com/klahssen/tester"
)

func TestNewFCT(t *testing.T) {
	te := tester.NewT(t)
	log := &log.Logger{}
	inSize := 3
	f1, _ := NewFC(inSize)
	lr := NewLr(0.5)
	//r1 := NewRandomDataset(42, inSize, 1000, 200, sum)
	tests := []struct {
		fc   *FC
		log  Logger
		lr   LrSource
		cost activation.F
		//training   Dataset
		//validation Dataset
		//test       Dataset
		maxIter uint
		tol     float64
		err     error
	}{
		{
			fc:  nil,
			log: nil,
			lr:  nil,
			//training:   nil,
			//validation: nil,
			//test:       nil,
			maxIter: 100,
			tol:     0.0,
			err:     fmt.Errorf("neural network is nil"),
		},
		{
			fc:  f1,
			log: nil,
			lr:  nil,
			//training:   nil,
			//validation: nil,
			//test:       nil,
			maxIter: 100,
			tol:     0.0,
			err:     fmt.Errorf("learning rate source is nil"),
		},
		{
			fc:   f1,
			log:  nil,
			lr:   lr,
			cost: activation.Iden(),
			//training:   nil,
			//validation: nil,
			//test:       nil,
			maxIter: 100,
			tol:     0.0,
			err:     nil,
		},
		/*
			{
				fc:         f1,
				log:        nil,
				lr:         lr,
				training:   nil,
				validation: nil,
				test:       nil,
				maxIter:    100,
				tol:        0.0,
				err:        fmt.Errorf("training set is empty"),
			},
			{
				fc:         f1,
				log:        nil,
				lr:         lr,
				training:   r1,
				validation: nil,
				test:       nil,
				maxIter:    100,
				tol:        0.0,
				err:        fmt.Errorf("test set is empty"),
			},
		*/
	}
	for ind, test := range tests {
		_, err := NewFCTrainer(test.fc, log, test.lr, test.maxIter, test.tol, test.cost)
		te.CheckError(ind, test.err, err)
	}
}
