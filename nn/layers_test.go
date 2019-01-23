package nn

import (
	"fmt"
	"testing"

	"github.com/klahssen/go-mat"
	"github.com/klahssen/nn/activation"

	"github.com/twiggg/tester"
)

func TestValidateLayerConfig(t *testing.T) {
	iden := activation.F{Func: func(x float64) float64 { return x }, Deriv: func(x float64) float64 { return 0.0 }}
	te := tester.New(t)
	tests := []struct {
		l   *LayerConfig
		err error
	}{
		{&LayerConfig{Size: -1}, fmt.Errorf("size must be >0")},
		{&LayerConfig{Size: 1}, fmt.Errorf("activation function is nil")},
		{&LayerConfig{Size: 1, F: activation.F{Func: func(x float64) float64 { return x }}}, fmt.Errorf("derivative of activation function is nil")},
		{&LayerConfig{Size: 1, F: iden}, nil},
		{nil, fmt.Errorf("level config is nil")},
	}
	for ind, test := range tests {
		te.CompareError(ind, test.err, test.l.Validate())
	}
}

func TestWxpb(t *testing.T) {
	te := tester.New(t)
	tests := []struct {
		w   *mat.M64
		x   *mat.M64
		b   *mat.M64
		res *mat.M64
		err error
	}{
		{
			w:   mat.NewM64(3, 3, []float64{1, 1, 1, 1, 1, 1, 1, 1, 1}),
			x:   mat.NewM64(3, 1, []float64{1, 2, 3}),
			b:   mat.NewM64(3, 1, []float64{0, 1, 2}),
			res: mat.NewM64(3, 1, []float64{6, 7, 8}),
			err: nil,
		},
	}
	for ind, test := range tests {
		res, err := wxpb(test.w, test.x, test.b)
		te.CompareError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "res", test.res, res)
		}
	}
}

func TestUpdateLayerData(t *testing.T) {
	iden := activation.F{Func: func(x float64) float64 { return x }, Deriv: func(x float64) float64 { return 0 }}
	//idenp := func(x float64) float64 { return 1 }
	te := tester.New(t)
	tests := []struct {
		l    *layer
		data []float64
		w    *mat.M64
		b    *mat.M64
		err  error
	}{
		{

			l:    newLayer(3, 2, "custom", nil, iden),
			data: []float64{1, 2, 3, 4, 5, 6, 7, 8},
			w:    mat.NewM64(2, 3, []float64{1, 2, 3, 5, 6, 7}),
			b:    mat.NewM64(2, 1, []float64{4, 8}),
			err:  nil,
		},
	}
	for ind, test := range tests {
		err := test.l.UpdateData(test.data)
		te.CompareError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "w", test.w, test.l.w)
			te.DeepEqual(ind, "b", test.b, test.l.b)
		}
	}
}

func TestComputeLayerWith(t *testing.T) {
	te := tester.New(t)
	db := activation.F{Func: func(x float64) float64 { return 2.0 * x }, Deriv: func(x float64) float64 { return 2.0 }}
	//dbp := func(x float64) float64 { return 2.0 }
	l1 := newLayer(3, 3, "custom", nil, db)
	l1.UpdateData([]float64{1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4})
	tests := []struct {
		l   *layer
		x   *mat.M64
		res *mat.M64
		err error
	}{
		{
			l:   l1,
			x:   mat.NewM64(3, 1, []float64{1, 1, 1}),
			res: mat.NewM64(3, 1, []float64{10, 12, 14}),
			err: nil,
		},
	}
	for ind, test := range tests {
		res, err := test.l.ComputeWith(test.x)
		te.CompareError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "res", test.res, res)
		}
	}
}
