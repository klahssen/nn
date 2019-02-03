package nn

import (
	"fmt"
	"testing"

	mat "github.com/klahssen/go-mat"
	"github.com/klahssen/nn/internal/activation"

	"github.com/klahssen/tester"
)

func TestNewFC(t *testing.T) {
	te := tester.NewT(t)
	tests := []struct {
		inSize     int
		keepStates bool
		ff         *FC
		err        error
	}{
		{
			inSize:     3,
			keepStates: true,
			ff: &FC{
				inSize:  3,
				outSize: 3,
				layers:  nil,
			},
			err: nil,
		},
	}
	for ind, test := range tests {
		ff, err := NewFC(test.inSize)
		te.CheckError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "network", test.ff, ff)
		}
	}
}

var mockFF = func(inSize int) *FC {
	f, _ := NewFC(3)
	return f
}

var mockFF2 = func(inSize int, configs []*LayerConfig) *FC {
	f, _ := NewFC(inSize)
	f.SetLayers(configs...) //should initialize states if keepStates
	//f.states = make([]*mat.M64, len(configs))
	return f
}

func TestFFSetLayers(t *testing.T) {
	f1 := activation.Sigmoid()
	//f1p := activation.DerivSigmoid
	te := tester.NewT(t)
	tests := []struct {
		ff      *FC
		configs []*LayerConfig
		exp     []*layer
		err     error
	}{
		{
			ff: mockFF(3),
			configs: []*LayerConfig{
				&LayerConfig{Size: 3, FuncType: activation.FuncTypeSigmoid, FuncParams: nil},
			},
			exp: []*layer{
				newLayer(3, 3, activation.FuncTypeSigmoid, nil, f1),
			},
			err: nil,
		},
		{
			ff:      mockFF(3),
			configs: []*LayerConfig{},
			exp:     nil,
			err:     fmt.Errorf("must have at least one layer"),
		},
		{
			ff: mockFF(3),
			configs: []*LayerConfig{
				&LayerConfig{Size: 3, FuncType: activation.FuncTypeSigmoid, FuncParams: nil},
			},
			exp: []*layer{
				newLayer(3, 3, activation.FuncTypeSigmoid, nil, f1),
			},
			err: nil,
		},
	}

	for ind, test := range tests {
		err := test.ff.SetLayers(test.configs...)
		te.CheckError(ind, test.err, err)
		if err == nil {
			l1 := len(test.exp)
			l2 := len(test.ff.layers)
			if l1 != l2 {
				t.Errorf("test %d: expected %d layers received %d", ind, l1, l2)
				continue
			}
			rs, cs := 0, 0
			for i, l := range test.ff.layers {
				le := test.exp[i]
				w := l.w
				we := le.w
				b := l.b
				be := le.b
				te.DeepEqual(ind, fmt.Sprintf("layer[%d].w", i), we, w)
				te.DeepEqual(ind, fmt.Sprintf("layer[%d].b", i), be, b)
				if l.keepState {
					rs, cs = l.state.Dims()
					if rs != l.outSize {
						t.Errorf("test %d: layer %d: expected %d rows received %d", ind, i, l.outSize, rs)
					}
					if rs != 1 {
						t.Errorf("test %d: layer %d: expected 1 colomn received %d", ind, i, cs)
					}
				}
			}

		}
	}
}

func TestFFGetState(t *testing.T) {
	te := tester.NewT(t)
	//ff2 := mockFF2(3, []*LayerConfig{{Size: 3, F: activation.Sigmoid()}})
	tests := []struct {
		ff       *FC
		layerInd int
		state    *mat.M64
		err      error
	}{
		{
			ff:       mockFF2(3, []*LayerConfig{{Size: 3, F: activation.Sigmoid()}}),
			layerInd: 0,
			state:    nil,
			err:      fmt.Errorf("state is nil"),
		},
		{
			ff:       mockFF2(3, []*LayerConfig{{Size: 3, KeepState: true, F: activation.Sigmoid()}}),
			layerInd: 0,
			state:    mat.NewM64(3, 1, nil),
			err:      nil,
		},
		{
			ff:       mockFF2(3, []*LayerConfig{{Size: 3, F: activation.Sigmoid()}}),
			layerInd: -1,
			state:    nil,
			err:      fmt.Errorf("layer index must be between 0 and 0"),
		},
		{
			ff:       mockFF2(3, []*LayerConfig{{Size: 3, F: activation.Sigmoid()}}),
			layerInd: 5,
			state:    nil,
			err:      fmt.Errorf("layer index must be between 0 and 0"),
		},
	}
	for ind, test := range tests {
		state, err := test.ff.GetState(test.layerInd)
		te.CheckError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "state", test.state, state)
		}
	}
}

func TestFCBackprop(t *testing.T) {
	te := tester.NewT(t)
	configs1 := []*LayerConfig{
		{KeepState: true, Size: 1, FuncType: "iden", FuncParams: nil},
	}
	f1 := mockFF2(2, configs1)
	f1.SetLayerData(0, []float64{1, 1, 0})
	tests := []struct {
		n        *FC
		lr       float64
		in       *mat.M64
		gradCost *mat.M64
		err      error
	}{
		{
			n:        f1,
			lr:       0.5,
			in:       mat.NewM64(2, 1, []float64{1, 1}),
			gradCost: mat.NewM64(1, 1, []float64{1}),
			err:      nil,
		},
	}
	for ind, test := range tests {
		test.n.FeedForward(test.in)
		/*
			for i := range test.n.layers {
				fmt.Printf("layer[%d]: %+v\n", i, test.n.layers[i])
			}
		*/
		err := test.n.Backprop(test.lr, test.in, test.gradCost)
		te.CheckError(ind, test.err, err)
	}
}