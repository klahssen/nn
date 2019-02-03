package nn

import (
	"fmt"

	"github.com/klahssen/nn/internal/activation"

	mat "github.com/klahssen/go-mat"
)

//ActivationFunc signature
//type ActivationFunc func(x float64) float64

//LayerConfig holds info to define a new layer
type LayerConfig struct {
	//InSize  int
	KeepState  bool
	Size       int
	FuncType   string
	FuncParams []float64
	F          activation.F
}

//Validate configuration
func (l *LayerConfig) Validate() error {
	if l == nil {
		return fmt.Errorf("level config is nil")
	}
	if l.Size <= 0 {
		return fmt.Errorf("size must be >0")
	}
	if l.FuncType == "" {
		if l.F.Func == nil {
			return fmt.Errorf("activation function is nil")
		}
		if l.F.Deriv == nil {
			return fmt.Errorf("derivative of activation function is nil")
		}
		l.FuncType = activation.FuncTypeCustom
	} else {
		f, err := activation.GetF(l.FuncType, l.FuncParams)
		if err != nil {
			return err
		}
		l.F = f
	}
	return nil
}

//newLayer returns a new Level
func newLayer(inSize int, outSize int, ftype string, fparams []float64, f activation.F) *layer {
	return &layer{
		inSize:  inSize,
		outSize: outSize,
		w:       mat.NewM64(outSize, inSize, nil),
		b:       mat.NewM64(outSize, 1, nil),
		ftype:   ftype,
		fparams: fparams,
		a:       f,
	}
}

//layer represents a layer of neurons, defined by Y=fn(w*X+b) where X is the input, Y the output,fn the activation function, W the weights matrix and b the bias.
type layer struct {
	keepState bool
	state     *mat.M64 //stores the output of each neuron (a)
	gradSig   *mat.M64 //stores the gradient of the activation
	inSize    int
	outSize   int
	w         *mat.M64
	b         *mat.M64
	ftype     string
	fparams   []float64
	a         activation.F
}

func (l *layer) Config() *LayerConfig {
	if l == nil {
		return nil
	}
	return &LayerConfig{
		Size:       l.outSize,
		KeepState:  l.keepState,
		FuncType:   l.ftype,
		FuncParams: l.fparams,
	}
}

func (l *layer) Validate() error {
	if l == nil {
		return fmt.Errorf("layer is nil")
	}
	if err := activation.ValidateFType(l.ftype); err != nil {
		return err
	}
	if l.ftype != activation.FuncTypeCustom {
		F, err := activation.GetF(l.ftype, l.fparams)
		if err != nil {
			return err
		}
		l.a = F
	}

	if l.a.Func == nil {
		return fmt.Errorf("activation function is nil")
	}
	if l.a.Deriv == nil {
		return fmt.Errorf("derivative of activation function is nil")
	}
	if l.inSize <= 0 {
		return fmt.Errorf("input size must be >0")
	}
	if l.outSize <= 0 {
		return fmt.Errorf("output size must be >0")
	}
	return nil
}

func (l *layer) IsUsable() error {
	if err := l.Validate(); err != nil {
		return err
	}
	if l.w == nil {
		return fmt.Errorf("weight matrix is nil")
	}
	if l.b == nil {
		return fmt.Errorf("bias vector is nil")
	}
	r, c := l.w.Dims()
	if r != l.outSize {
		return fmt.Errorf("weight matrix should have %d rows not %d", l.outSize, r)
	}
	if c != l.inSize {
		return fmt.Errorf("weight matrix should have %d colomns not %d", l.inSize, c)
	}
	r, c = l.b.Dims()
	if r != l.outSize {
		return fmt.Errorf("bias vector should have %d rows not %d", l.outSize, r)
	}
	if c != 1 {
		return fmt.Errorf("bias vector should have 1 colomns not %d", c)
	}
	if l.keepState {
		l.state = mat.NewM64(r, c, nil)
		l.gradSig = mat.NewM64(r, c, nil)
	} else {
		l.state = nil
	}
	return nil
}

func (l *layer) dataSize() int {
	//should be the size of w + size of b
	//which is out*in +out*1=out*(in+1)
	if l == nil {
		return 0
	}
	return l.outSize * (l.inSize + 1)
}

//data should have outxin values for w then outx1 values for b
func (l *layer) UpdateData(data []float64) error {
	if l == nil {
		return fmt.Errorf("layer is nil")
	}
	size := l.dataSize()
	if len(data) != size {
		return fmt.Errorf("expected %d values", size)
	}
	//each group of <in+1> values is <in> values for 1 line of w and <1> value for b
	/*
		ind := 0
		i, j := 0, 0
		for i = 0; i < l.outSize; i++ {
			for j = 0; j < l.inSize; j++ {
				l.w.Set(i, j, data[ind])
				ind++
				if j == l.inSize-1 {
					l.b.Set(i, 0, data[ind])
					ind++
				}
			}
		}
	*/
	lim := l.outSize * l.inSize
	var err error
	if err = l.w.SetData(data[:lim]); err != nil {
		return fmt.Errorf("failed to update weights: %s", err.Error())
	}
	if err = l.b.SetData(data[lim:]); err != nil {
		return fmt.Errorf("failed to update bias: %s", err.Error())
	}
	return nil
}

//FeedForward computes the activation output from the input
func (l *layer) FeedForward(input *mat.M64) (*mat.M64, error) {
	if l == nil {
		return nil, fmt.Errorf("layer is nil")
	}
	res, err := wxpb(l.w, input, l.b)
	if err != nil {
		return nil, err
	}
	//compute sigmaPrimes during forward pass, preparing for backprop (backward pass)
	if l.keepState {
		l.gradSig, err = mat.MapElem(res, l.a.Deriv)
		if err != nil {
			return nil, fmt.Errorf("failed to compute sigPrimes: %s", err.Error())
		}
	}
	res, err = mat.MapElem(res, l.a.Func)
	if err == nil && l.keepState {
		l.state = res
	}
	return res, err
}

//init sets all weights and bias to 1
func (l *layer) init() {
	if l == nil {
		return
	}
	n := l.dataSize()
	data := make([]float64, n)
	for i := range data {
		data[i] = 1.0
	}
	l.UpdateData(data)
}

/*
Backprop propagates the gradient from layer l+1 to layer l, and updates inner w and b in layer l
gradCost is the gradiant vector of the cost depending on the activation (backpropagated from layer l+1 if not output layer)
gradSig is the derivative vector of the activation (not used if output layer)
w is the weight matrix in layer l+1 (not used if output layer)
in is the input (from layer l-1)
*/
func (l *layer) Backprop(lr float64, in, gradCost, gradSig, w *mat.M64, isOutputLayer bool) (*mat.M64, error) {
	if l == nil {
		return nil, fmt.Errorf("layer is nil")
	}
	if l.w == nil {
		return nil, fmt.Errorf("weight matrix is nil")
	}
	if l.b == nil {
		return nil, fmt.Errorf("bias vector is nil")
	}
	if lr <= 0 || lr > 1.0 {
		return nil, fmt.Errorf("learning rate must be in range ]0;1]")
	}
	if gradCost == nil {
		return nil, fmt.Errorf("cost gradient vector is nil")
	}

	if in == nil {
		return nil, fmt.Errorf("local input vector is nil")
	}
	var cprime *mat.M64
	var err error
	//if isOutputLayer, no downstream dependency (root of the back-propagation)
	if !isOutputLayer {
		//backpropagate downstream costPrimes to compute local costPrimes
		if gradSig == nil {
			return nil, fmt.Errorf("activation gradient vector is nil")
		}
		if w == nil {
			return nil, fmt.Errorf("weight matrix is nil")
		}
		/*
			r, c := gradSig.Dims()
			fmt.Printf("gradSig r=%d c=%d\n", r, c)
			r, c = gradCost.Dims()
			fmt.Printf("gradCost r=%d c=%d\n", r, c)
		*/
		//layer l+1 outx1
		cprime, err = mat.MulElem(gradSig, gradCost)
		if err != nil {
			return nil, fmt.Errorf("failed to compute gradSig*gradCost: %s", err.Error())
		}
		//layer l+1 inx1 = layer l outx1
		w.Transpose()
		defer w.DeTranspose()
		//r, c = w.Dims()
		//fmt.Printf("wT r=%d c=%d\n", r, c)
		cprime, err = mat.Mul(w, cprime)
		if err != nil {
			return nil, fmt.Errorf("failed to compute wT*gradSig*gradCost: %s", err.Error())
		}
		//r, c = cprime.Dims()
		//fmt.Printf("cprime r=%d c=%d\n", r, c)

	} else {
		cprime = gradCost
	}
	//delta is outx1
	/*
		r, c := cprime.Dims()
		fmt.Printf("cprime.Dims r=%d c=%d\n", r, c)
		r, c = l.gradSig.Dims()
		fmt.Printf("l.gradSig.Dims r=%d c=%d\n", r, c)
	*/
	//update weights and bias
	//gradB is outx1
	gradB, err := mat.MulElem(l.gradSig, cprime)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient of bias vector: %s", err.Error())
	}
	if err = gradB.MapElem(func(x float64) float64 { return lr * x }); err != nil {
		return nil, fmt.Errorf("failed to multiply gradient by learning rate: %s", err.Error())
	}
	//gradW is outxin
	in.Transpose()
	defer in.DeTranspose()
	//r, c := in.Dims()
	//r, c = gradB.Dims()
	gradW, err := mat.Mul(gradB, in)
	if err != nil {
		return nil, fmt.Errorf("failed to compute gradient of weight matrix: %s", err.Error())
	}
	if err = l.w.Sub(gradW); err != nil {
		return nil, fmt.Errorf("failed to update weights: %s", err.Error())
	}
	if err = l.b.Sub(gradB); err != nil {
		return nil, fmt.Errorf("failed to update bias: %s", err.Error())
	}
	return cprime, nil
}

//wxpb computes the dot product of w and x then adds b
func wxpb(w, x, b *mat.M64) (*mat.M64, error) {
	res, err := mat.Mul(w, x)
	if err != nil {
		return nil, fmt.Errorf("w*x failed: %s", err.Error())
	}
	res, err = mat.Add(res, b)
	if err != nil {
		return nil, fmt.Errorf("w*x +b failed: %s", err.Error())
	}

	return res, nil
}
