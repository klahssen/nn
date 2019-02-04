package nn

import (
	"fmt"

	"github.com/klahssen/go-mat"
)

//FC represents a simple fully connected feed forward neural network
type FC struct {
	inSize  int
	outSize int
	layers  []*layer
}

//NewFC returns a new instance of Fully Connected FeedForward Neural Network, with no layers
func NewFC(inSize int) (*FC, error) {
	if inSize < 1 {
		return nil, fmt.Errorf("minimum input size is 1")
	}
	ff := &FC{
		inSize:  inSize,
		outSize: inSize,
	}
	return ff, nil
}

//init sets all weights and bias to 1 in each layer
func (ff *FC) init() {
	if ff == nil {
		return
	}
	for i := range ff.layers {
		ff.layers[i].init()
	}
}

//SetLayers sets neuron layers connected via w,b,fn. Must have at least 1 layer
func (ff *FC) SetLayers(configs ...*LayerConfig) error {
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
			lay.keepState = true
		}
		layers[i] = lay
		prevSize = l.Size
	}
	ff.layers = layers

	return nil
}

//SetLayerData sets W and B matrices in layer at layerInd
func (ff *FC) SetLayerData(layerInd int, data []float64) error {
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
func (ff *FC) validate() error {
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
func (ff *FC) Info() {
	if err := ff.validate(); err != nil {
		panic(err)
	}
	neurons := 0
	fmt.Printf("\n======================= DEFINITION =======================\nInfo about this FeedForward Neural Net:\nlayer(s): %d\n\n", len(ff.layers))
	//defs := make([]*LayerConfig, len(ff.layers))
	for i, l := range ff.layers {
		neurons += l.outSize
		//defs[i] = l.Config()
		fmt.Printf("- Layer %d:\nneuron(s): %d\nactivation type: '%s'\nactivation params: %v\n\n", i+1, l.outSize, l.ftype, l.fparams)
	}

	fmt.Printf("Total: %d neuron(s)\n=========================================================\n\n\n", neurons)
}

//FeedForward feeds data forward from input, returns output layer's state
func (ff *FC) FeedForward(input *mat.M64) (*mat.M64, error) {
	in := input
	var out *mat.M64
	var err error
	for i, l := range ff.layers {
		out, err = l.FeedForward(in)
		if err != nil {
			return nil, fmt.Errorf("layer[%d]: %s", i, err.Error())
		}
		in = out
	}
	return out, nil
}

//Backprop the cost gradient
func (ff *FC) Backprop(lr float64, in, gradCost *mat.M64) error {
	if ff == nil {
		return fmt.Errorf("network is nil")
	}
	if in == nil {
		return fmt.Errorf("input is nil")
	}
	if gradCost == nil {
		return fmt.Errorf("cost gradient is nil")
	}
	var err error
	n := len(ff.layers)
	ind := 0
	var next *layer
	var input *mat.M64
	for i := range ff.layers {
		ind = (n - 1) - i
		if ind == 0 {
			input = in
		} else {
			input = ff.layers[ind-1].state
		}
		/*
			fmt.Printf("layers[%d]\n", ind)
			fmt.Printf("	state: %+v\n", ff.layers[ind].state.GetData())
			fmt.Printf("	b:%+v\n", ff.layers[ind].b.GetData())
			fmt.Printf("	w:%+v\n", ff.layers[ind].w.GetData())
			fmt.Printf("	gradSig: %+v\n", ff.layers[ind].gradSig.GetData())
		*/
		if i == 0 {
			//output layer
			//fmt.Printf("	params: gradCost:%+v gradSig:%+v w:%+v\n", gradCost.GetData(), nil, nil)
			gradCost, err = ff.layers[ind].Backprop(lr, input, gradCost, nil, nil, true)
			if err != nil {
				return fmt.Errorf("layer %d: %s", ind, err.Error())
			}
			next = ff.layers[ind]
			//fmt.Printf("	new b:%+v\n", ff.layers[ind].b.GetData())
			//fmt.Printf("	new w:%+v\n", ff.layers[ind].w.GetData())
		} else {
			//fmt.Printf("params: gradCost:%+v gradSig:%+v w:%+v\n", gradCost.GetData(), next.gradSig.GetData(), next.w.GetData())
			gradCost, err = ff.layers[ind].Backprop(lr, input, gradCost, next.gradSig, next.w, false)
			if err != nil {
				return fmt.Errorf("layer %d: %s", ind, err.Error())
			}
			next = ff.layers[ind]
			//fmt.Printf("	new b:%+v\n", ff.layers[ind].b.GetData())
			//fmt.Printf("	new w:%+v\n", ff.layers[ind].w.GetData())
		}
	}
	return nil
}

//GetState returns the output values of a layer if keepStates==true or an error
func (ff *FC) GetState(layerInd int) (*mat.M64, error) {
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
