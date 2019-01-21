package nn

import (
	"math/rand"
	"testing"

	"github.com/twiggg/tester"
)

func TestSelectDrops(t *testing.T) {
	te := tester.New(t)
	tests := []struct {
		r         rand.Source
		dropSize  int
		fleetSize int
		res       map[int]struct{}
	}{
		{
			r: rand.NewSource(42), dropSize: 5, fleetSize: 10,
			res: map[int]struct{}{0: struct{}{}, 9: struct{}{}, 7: struct{}{}, 5: struct{}{}, 1: struct{}{}},
		},
	}
	for ind, test := range tests {
		res := selectDrops(test.r, test.dropSize, test.fleetSize)
		te.DeepEqual(ind, "res", test.res, res)
	}
}
