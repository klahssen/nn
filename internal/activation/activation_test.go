package activation

import (
	"fmt"
	"strings"
	"testing"

	"github.com/klahssen/tester"
)

func TestGetF(t *testing.T) {
	te := tester.NewT(t)
	tests := []struct {
		ftype  string
		params []float64
		err    error
	}{
		{ftype: "custom", err: fmt.Errorf("invalid activation function type 'custom': expected one of [%s]", strings.Join(getValidFTypes(), ", "))},
		{ftype: "okok", err: fmt.Errorf("invalid activation function type 'okok': expected one of [%s]", strings.Join(getValidFTypes(), ", "))},
		{ftype: "relu", params: nil, err: nil},
		{ftype: "elu", params: nil, err: fmt.Errorf("expected 1 parameter(s) for func 'elu'")},
		{ftype: "elu", params: []float64{1.0}, err: nil},
	}
	var err error
	for ind, test := range tests {
		_, err = GetF(test.ftype, test.params)
		te.CheckError(ind, test.err, err)
	}
}

func TestNewPower(t *testing.T) {
	te := tester.NewT(t)
	tests := []struct {
		n    uint
		coef float64
		x    float64
		res  float64
	}{
		{2, 1.0, 2.0, 4.0},
		{3, 1.0, 2.0, 8.0},
		{2, 2.0, 2.0, 8.0},
	}
	for ind, test := range tests {
		f := newPower(test.coef, test.n)
		res := f(test.x)
		te.DeepEqual(ind, "res", test.res, res)
	}
}
