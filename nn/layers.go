package nn

import (
	"fmt"

	mat "github.com/klahssen/go-mat"
	"github.com/klahssen/nn/activation"
)

//ActivationFunc signature
//type ActivationFunc func(x float64) float64

//LayerConfig holds info to define a new layer
type LayerConfig struct {
	//InSize  int
	Size       int
	FuncType   string
	FuncParams []float64
	F          activation.F
}

//Validate checks configuration data
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
		a:       f,
	}
}

//layer represents a layer of neurons, defined by Y=fn(w*X+b) where X is the input, Y the output,fn the activation function, W the weights matrix and b the bias.
type layer struct {
	inSize  int
	outSize int
	w       *mat.M64
	b       *mat.M64
	ftype   string
	fparams []float64
	a       activation.F
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
	return nil
}

func (l *layer) dataSize() int {
	//should be the size of w + size of b
	//which is out*in +out*1=out*(in+1)
	if l == nil {
		return 0
	}
	return l.outSize * (l.inSize + 1) //l.w.Size() + l.b.Size()
}

func (l *layer) UpdateData(data []float64) error {
	if l == nil {
		return fmt.Errorf("layer is nil")
	}
	size := l.dataSize()
	if len(data) != size {
		return fmt.Errorf("expected %d values", size)
	}
	//each group of <in+1> values is <in> values for 1 line of w and <1> value for b
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
	return nil
}

func (l *layer) ComputeWith(input *mat.M64) (*mat.M64, error) {
	res, err := wxpb(l.w, input, l.b)
	if err != nil {
		return nil, err
	}
	return mat.MapElem(res, l.a.Func)
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
