package nn

import (
	"fmt"

	"github.com/klahssen/go-mat"
)

//FFN represents a simple feed forward neural network
type FFN struct {
	inSize     int
	outSize    int
	layers     []*layer
	states     []*mat.M64
	keepStates bool
}

//NewFFN returns a new instance of FeedForward Neural Network, with no layers
func NewFFN(inSize int, keepStates bool) (*FFN, error) {
	if inSize < 1 {
		return nil, fmt.Errorf("minimum input size is 1")
	}
	ff := &FFN{
		inSize:     inSize,
		outSize:    inSize,
		keepStates: keepStates,
	}
	return ff, nil
}

//SetLayers sets neuron layers connected via w,b,fn. Must have at least 1 layer
func (ff *FFN) SetLayers(configs ...*LayerConfig) error {
	var err error
	n := len(configs)
	if n < 1 {
		return fmt.Errorf("must have at least one layer")
	}
	n2 := 0
	if ff.keepStates {
		n2 = n
	}
	prevSize := ff.inSize
	layers := make([]*layer, n)

	for i, l := range configs {
		if err = l.Validate(); err != nil {
			return fmt.Errorf("configs[%d]: %s", i, err.Error())
		}
		layers[i] = newLayer(prevSize, l.Size, l.FuncType, l.FuncParams, l.F)
		prevSize = l.Size
	}
	ff.layers = layers
	ff.states = make([]*mat.M64, n2)
	return nil
}

//Feed feeds data forward from input, returns output layer's state
func (ff *FFN) Feed(input *mat.M64) (*mat.M64, error) {
	in := input
	var out *mat.M64
	if ff.keepStates {
		ff.states = make([]*mat.M64, len(ff.layers))
	}
	for i, l := range ff.layers {
		out, err := l.ComputeWith(in)
		if err != nil {
			return nil, fmt.Errorf("layer[%d]: %s", i, err.Error())
		}
		if ff.keepStates {
			ff.states[i] = out
		}
		in = out
	}
	return out, nil
}

//GetState returns the output values of a layer if keepStates==true or an error
func (ff *FFN) GetState(layerInd int) (*mat.M64, error) {
	if ff == nil {
		return nil, fmt.Errorf("network is nil")
	}
	l := len(ff.states)
	if l == 0 {
		return nil, fmt.Errorf("no states")
	}
	if layerInd < 0 || layerInd > l-1 {
		return nil, fmt.Errorf("layer index must be between %d and %d", 0, l-1)
	}
	s := ff.states[layerInd]
	if s == nil {
		return nil, fmt.Errorf("state is nil")
	}
	return s, nil
}
