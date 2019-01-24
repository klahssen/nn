package nn

import (
	"fmt"

	"github.com/klahssen/go-mat"
)

//FFN represents a simple feed forward neural network
type FFN struct {
	inSize  int
	outSize int
	layers  []*layer
}

//NewFFN returns a new instance of FeedForward Neural Network, with no layers
func NewFFN(inSize int) (*FFN, error) {
	if inSize < 1 {
		return nil, fmt.Errorf("minimum input size is 1")
	}
	ff := &FFN{
		inSize:  inSize,
		outSize: inSize,
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

	prevSize := ff.inSize
	layers := make([]*layer, n)

	for i, l := range configs {
		if err = l.Validate(); err != nil {
			return fmt.Errorf("configs[%d]: %s", i, err.Error())
		}
		lay := newLayer(prevSize, l.Size, l.FuncType, l.FuncParams, l.F)
		if l.KeepState {
			lay.state = mat.NewM64(l.Size, 1, nil)
		}
		layers[i] = lay
		prevSize = l.Size
	}
	ff.layers = layers

	return nil
}

//SetLayerData sets W and B matrices in layer at layerInd
func (ff *FFN) SetLayerData(layerInd int, data []float64) error {
	if ff == nil {
		return fmt.Errorf("network is nil")
	}
	l := len(ff.layers)
	if layerInd < 0 || layerInd > l-1 {
		return fmt.Errorf("layer index must be between %d and %d", 0, l-1)
	}
	return ff.layers[layerInd].UpdateData(data)
}

//Validate checks if network is runnable
func (ff *FFN) Validate() error {
	if ff == nil {
		return fmt.Errorf("network is nil")
	}
	if ff.inSize <= 0 {
		return fmt.Errorf("network input size is <=0")
	}
	if ff.outSize <= 0 {
		return fmt.Errorf("network output size is <=0")
	}
	if len(ff.layers) == 0 {
		return fmt.Errorf("network has no layers")
	}
	var err error
	for i, l := range ff.layers {
		if err = l.Validate(); err != nil {
			return fmt.Errorf("layers[%d]: %s", i, err.Error())
		}
	}
	return nil
}

//Info prints a summary of the network's definition
func (ff *FFN) Info() {
	if err := ff.Validate(); err != nil {
		panic(err)
	}
	neurons := 0
	fmt.Printf("\n======================= DEFINITION =======================\nInfo about this FeedForward Neural Net:\nlayer(s): %d\n\n", len(ff.layers))
	//defs := make([]*LayerConfig, len(ff.layers))
	for i, l := range ff.layers {
		neurons += l.outSize
		//defs[i] = l.Config()
		fmt.Printf("- Layer %d:\nneurons: %d\nactivation type: '%s'\nactivation params: %v\n\n", i+1, l.outSize, l.ftype, l.fparams)
	}

	fmt.Printf("Total: %d neurons\n=========================================================\n\n\n", neurons)
}

//Feed feeds data forward from input, returns output layer's state
func (ff *FFN) Feed(input *mat.M64) (*mat.M64, error) {
	in := input
	var out *mat.M64
	var err error
	for i, l := range ff.layers {
		out, err = l.ComputeWith(in)
		if err != nil {
			return nil, fmt.Errorf("layer[%d]: %s", i, err.Error())
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
	l := len(ff.layers)
	if layerInd < 0 || layerInd > l-1 {
		return nil, fmt.Errorf("layer index must be between %d and %d", 0, l-1)
	}
	s := ff.layers[layerInd].state
	if s == nil {
		return nil, fmt.Errorf("state is nil")
	}
	return s, nil
}
