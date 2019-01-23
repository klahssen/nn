package activation

import (
	"fmt"
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
		{ftype: "custom", err: fmt.Errorf("invalid activation function type 'custom': expected one of [custom, elu, leaky_relu, relu, sig, tanh]")},
		{ftype: "okok", err: fmt.Errorf("invalid activation function type 'okok': expected one of [custom, elu, leaky_relu, relu, sig, tanh]")},
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
