package generic

import (
	"testing"

	"github.com/klahssen/tester"
)

func TestSum(t *testing.T) {
	te := tester.NewT(t)
	tests := []struct {
		x   []float64
		res float64
	}{
		{x: []float64{-1.0, -1.0}, res: -2.0},
		{x: []float64{-1.0, 1.0}, res: 0.0},
		{x: []float64{1.0, 2.0}, res: 3.0},
		{x: []float64{0.0, 0.0}, res: 0.0},
	}

	for ind, test := range tests {
		te.DeepEqual(ind, "res", test.res, sum(test.x))

	}
}
