package nn

import (
	"fmt"
	"testing"

	"github.com/klahssen/nn/internal/activation"

	"github.com/klahssen/go-mat"

	"github.com/klahssen/tester"
)

func TestValidateLayerConfig(t *testing.T) {
	iden := activation.F{Func: func(x float64) float64 { return x }, Deriv: func(x float64) float64 { return 0.0 }}
	te := tester.NewT(t)
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
		te.CheckError(ind, test.err, test.l.Validate())
	}
}

func TestWxpb(t *testing.T) {
	te := tester.NewT(t)
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
		te.CheckError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "res", test.res, res)
		}
	}
}

func TestUpdateLayerData(t *testing.T) {
	iden := activation.F{Func: func(x float64) float64 { return x }, Deriv: func(x float64) float64 { return 0 }}
	//idenp := func(x float64) float64 { return 1 }
	te := tester.NewT(t)
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
			w:    mat.NewM64(2, 3, []float64{1, 2, 3, 4, 5, 6}),
			b:    mat.NewM64(2, 1, []float64{7, 8}),
			err:  nil,
		},
	}
	for ind, test := range tests {
		err := test.l.UpdateData(test.data)
		te.CheckError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "w", test.w, test.l.w)
			te.DeepEqual(ind, "b", test.b, test.l.b)
		}
	}
}

func TestComputeLayerWith(t *testing.T) {
	te := tester.NewT(t)
	db := activation.F{Func: func(x float64) float64 { return 2.0 * x }, Deriv: func(x float64) float64 { return 2.0 }}
	//dbp := func(x float64) float64 { return 2.0 }
	l1 := newLayer(3, 3, "custom", nil, db)
	l1.UpdateData([]float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4})
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
		res, err := test.l.FeedForward(test.x)
		te.CheckError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "res", test.res, res)
		}
	}
}

func TestLayerBackprop(t *testing.T) {
	te := tester.NewT(t)
	l1 := newLayer(2, 1, "iden", nil, activation.Iden())
	l1.keepState = true
	l1.UpdateData([]float64{1, 2, 0})
	l2 := newLayer(2, 2, "iden", nil, activation.Iden())
	l2.keepState = true
	l2.UpdateData([]float64{1, 2, 1, 2, 0, 0})
	tests := []struct {
		l             *layer
		lr            float64
		w             *mat.M64
		in            *mat.M64
		gradSig       *mat.M64
		gradCost      *mat.M64
		isOutputLayer bool
		cprimea       *mat.M64
		newW          *mat.M64
		newB          *mat.M64
		err           error
	}{
		{
			l:             l1,
			lr:            0.5,
			w:             nil,
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       nil,
			gradCost:      mat.NewM64(1, 1, []float64{0.5}),
			isOutputLayer: true,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          mat.NewM64(1, 2, []float64{0.75, 1.5}),
			newB:          mat.NewM64(1, 1, []float64{-0.25}),
			err:           nil,
		},
		{
			l:             l1,
			lr:            0.5,
			w:             nil,
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       nil,
			gradCost:      mat.NewM64(1, 1, []float64{0.5}),
			isOutputLayer: false,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          nil,
			newB:          nil,
			err:           fmt.Errorf("activation gradient vector is nil"),
		},
		{
			l:             l1,
			lr:            0.5,
			w:             nil,
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       mat.NewM64(1, 1, []float64{1}),
			gradCost:      nil,
			isOutputLayer: false,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          nil,
			newB:          nil,
			err:           fmt.Errorf("cost gradient vector is nil"),
		},
		{
			l:             nil,
			lr:            0.5,
			w:             nil,
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       mat.NewM64(1, 1, []float64{1}),
			gradCost:      nil,
			isOutputLayer: false,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          nil,
			newB:          nil,
			err:           fmt.Errorf("layer is nil"),
		},
		{
			l:             l1,
			lr:            -0.5,
			w:             nil,
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       mat.NewM64(1, 1, []float64{1}),
			gradCost:      nil,
			isOutputLayer: false,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          nil,
			newB:          nil,
			err:           fmt.Errorf("learning rate must be in range ]0;1]"),
		},
		{
			l:             l1,
			lr:            0.5,
			w:             nil,
			in:            nil,
			gradSig:       mat.NewM64(1, 1, []float64{1}),
			gradCost:      mat.NewM64(1, 1, []float64{0.5}),
			isOutputLayer: false,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          nil,
			newB:          nil,
			err:           fmt.Errorf("local input vector is nil"),
		},
		{
			l:             l1,
			lr:            0.5,
			w:             nil,
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       mat.NewM64(1, 1, []float64{1}),
			gradCost:      mat.NewM64(1, 1, []float64{0.5}),
			isOutputLayer: false,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          nil,
			newB:          nil,
			err:           fmt.Errorf("weight matrix is nil"),
		},
		{
			l:             l1,
			lr:            0.5,
			w:             nil,
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       mat.NewM64(1, 1, []float64{1}),
			gradCost:      mat.NewM64(1, 1, []float64{0.5}),
			isOutputLayer: false,
			cprimea:       mat.NewM64(1, 1, []float64{0.5}),
			newW:          nil,
			newB:          nil,
			err:           fmt.Errorf("weight matrix is nil"),
		},
		{
			l:             l2,
			lr:            0.5,
			w:             mat.NewM64(1, 2, []float64{1, 2}),
			in:            mat.NewM64(2, 1, []float64{1, 2}),
			gradSig:       mat.NewM64(1, 1, []float64{1}),
			gradCost:      mat.NewM64(1, 1, []float64{0.5}),
			isOutputLayer: false,
			cprimea:       mat.NewM64(2, 1, []float64{0.5, 1}),
			newW:          mat.NewM64(2, 2, []float64{0.75, 1.5, 0.5, 1}),
			newB:          mat.NewM64(2, 1, []float64{-0.25, -0.5}),
			err:           nil,
		},
	}

	for ind, test := range tests {
		//fmt.Printf("test %d\n",ind)
		test.l.FeedForward(test.in)
		/*if err != nil {
			t.Errorf("test %d: failed to feed input to the layer: %s", ind, err.Error())
			continue
		}*/
		gradCost, err := test.l.Backprop(test.lr, test.in, test.gradCost, test.gradSig, test.w, test.isOutputLayer)
		te.CheckError(ind, test.err, err)
		if err != nil {
			continue
		}
		te.DeepEqual(ind, "cost gradient", test.cprimea, gradCost)
		te.DeepEqual(ind, "new w", test.newW, test.l.w)
		te.DeepEqual(ind, "new b", test.newB, test.l.b)
	}
}
