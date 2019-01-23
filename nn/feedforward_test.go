package nn

import (
	"fmt"
	"testing"

	mat "github.com/klahssen/go-mat"
	"github.com/klahssen/nn/activation"

	"github.com/twiggg/tester"
)

func TestNewFF(t *testing.T) {
	te := tester.New(t)
	tests := []struct {
		inSize     int
		keepStates bool
		ff         *FFN
		err        error
	}{
		{
			inSize:     3,
			keepStates: true,
			ff: &FFN{
				inSize:     3,
				outSize:    3,
				keepStates: true,
				layers:     nil,
				states:     nil,
			},
			err: nil,
		},
	}
	for ind, test := range tests {
		ff, err := NewFFN(test.inSize, test.keepStates)
		te.CompareError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "network", test.ff, ff)
		}
	}
}

var getFF = func(inSize int, keepStates bool) *FFN {
	f, _ := NewFFN(3, keepStates)
	return f
}

var getFF2 = func(inSize int, keepStates bool, configs []*LayerConfig) *FFN {
	f, _ := NewFFN(3, keepStates)
	f.SetLayers(configs...) //should initialize states if keepStates
	//f.states = make([]*mat.M64, len(configs))
	return f
}

func TestFFSetLayers(t *testing.T) {
	f1 := activation.Sigmoid()
	//f1p := activation.DerivSigmoid
	te := tester.New(t)
	tests := []struct {
		ff      *FFN
		configs []*LayerConfig
		exp     []*layer
		err     error
	}{
		{
			ff: getFF(3, false),
			configs: []*LayerConfig{
				&LayerConfig{Size: 3, FuncType: activation.FuncTypeSigmoid, FuncParams: nil},
			},
			exp: []*layer{
				newLayer(3, 3, activation.FuncTypeSigmoid, nil, f1),
			},
			err: nil,
		},
		{
			ff:      getFF(3, false),
			configs: []*LayerConfig{},
			exp:     nil,
			err:     fmt.Errorf("must have at least one layer"),
		},
		{
			ff: getFF(3, true),
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
		te.CompareError(ind, test.err, err)
		if err == nil {
			l1 := len(test.exp)
			l2 := len(test.ff.layers)
			if l1 != l2 {
				t.Errorf("test %d: expected %d layers received %d", ind, l1, l2)
				continue
			}
			for i, l := range test.ff.layers {
				le := test.exp[i]
				w := l.w
				we := le.w
				b := l.b
				be := le.b
				te.DeepEqual(ind, fmt.Sprintf("layer[%d].w", i), we, w)
				te.DeepEqual(ind, fmt.Sprintf("layer[%d].b", i), be, b)
			}
			nstates := 0
			if test.ff.keepStates {
				nstates = len(test.ff.layers)
			}
			n2 := len(test.ff.states)
			if nstates != n2 {
				t.Errorf("test %d: expected %d states, has %d", ind, nstates, n2)
			}
			//fmt.Printf("states:\n%+v\n", test.ff.states)
		}
	}
}

func TestFFGetState(t *testing.T) {
	te := tester.New(t)
	ff2 := getFF2(3, true, []*LayerConfig{{Size: 3, F: activation.Sigmoid()}})
	ff2.states = []*mat.M64{mat.NewM64(3, 3, nil)}
	tests := []struct {
		ff       *FFN
		layerInd int
		state    *mat.M64
		err      error
	}{
		{
			ff:       getFF2(3, true, []*LayerConfig{{Size: 3, F: activation.Sigmoid()}}),
			layerInd: 0,
			state:    mat.NewM64(3, 3, nil),
			err:      fmt.Errorf("state is nil"),
		},
		{
			ff:       ff2,
			layerInd: 0,
			state:    mat.NewM64(3, 3, nil),
			err:      nil,
		},
		{
			ff:       ff2,
			layerInd: -1,
			state:    mat.NewM64(3, 3, nil),
			err:      fmt.Errorf("layer index must be between 0 and 0"),
		},
		{
			ff:       ff2,
			layerInd: 5,
			state:    mat.NewM64(3, 3, nil),
			err:      fmt.Errorf("layer index must be between 0 and 0"),
		},
	}
	for ind, test := range tests {
		state, err := test.ff.GetState(test.layerInd)
		te.CompareError(ind, test.err, err)
		if err == nil {
			te.DeepEqual(ind, "state", test.state, state)
		}
	}
}
