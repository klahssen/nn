package nn

import (
	"encoding/json"
	"fmt"
	"io/ioutil"

	"github.com/klahssen/go-mat"
	"github.com/klahssen/nn/internal/activation"
)

//Perceptron is the simplest neuron, representing a function P. It applies an activation function f to s which is the weighted sum of its inputs + bias: output=f(w*x+b). the multiplication here is a dot product and wx+b is a scalar
type Perceptron struct {
	inSize int
	w      *mat.M64 //size 1*inSize
	f      activation.F
	cost   activation.F
	b      float64
	alpha  float64 //learning rate
	s      float64
	a      float64
}

//Perceptron is the simplest neuron, representing a function P. It applies an activation function f to s which is the weighted sum of its inputs + bias: output=f(w*x+b). the multiplication here is a dot product and wx+b is a scalar
type publicPerceptron struct {
	InSize int          `json:"in_size"`
	W      []float64    `json:"w"` //size 1*inSize
	F      activation.F `json:"-"`
	Cost   activation.F `json:"-"`
	B      float64      `json:"b"`
	Alpha  float64      `json:"alpha"` //learning rate
	S      float64      `json:"s"`
	A      float64      `json:"a"`
}

func (p *Perceptron) export() *publicPerceptron {
	return &publicPerceptron{InSize: p.inSize, W: p.w.GetData(), F: p.f, Cost: p.cost, B: p.b, Alpha: p.alpha, S: p.s, A: p.a}
}

//JSON stores the neuron's definition in  a json file
func (p *Perceptron) JSON(filename string) error {
	b, err := json.Marshal(p.export())
	if err != nil {
		return err
	}
	if err = ioutil.WriteFile(filename, b, 0666); err != nil {
		return err
	}
	fmt.Printf("Perceptron Configuration:\n%s\n", string(b))
	return nil
}

//Compute the P(x)
func (p *Perceptron) Compute(x *mat.M64) (float64, error) {
	res, err := mat.Mul(p.w, x)
	if err != nil {
		return 0.0, err
	}
	fn := func(x float64) float64 {
		return p.f.Func(x + p.b)
	}
	//fmt.Printf("W*X= %v\n", res.GetData())
	p.s = res.AtInd(0) + p.b
	if err = res.MapElem(fn); err != nil {
		return 0.0, err
	}
	p.a = res.AtInd(0)
	return p.a, nil
}

//UpdateCoefs updates inner weights and bias
func (p *Perceptron) UpdateCoefs(data []float64) error {
	nw := p.w.Size()
	size := nw + 1
	if len(data) != size {
		return fmt.Errorf("data should count 1 bias + %d weights", p.w.Size())
	}
	p.b = data[0]
	if err := p.w.SetData(data[1:]); err != nil {
		return fmt.Errorf("can not update weights: %s", err.Error())
	}
	return nil
}

//Validate checks if everything is usable
func (p *Perceptron) Validate() error {
	if p == nil {
		return fmt.Errorf("perceptron is nil")
	}
	if p.w == nil {
		return fmt.Errorf("weight matrix is nil")
	}
	if p.f.Func == nil {
		return fmt.Errorf("activation function is nil")
	}
	if p.f.Deriv == nil {
		return fmt.Errorf("activation derivative is nil")
	}
	if p.cost.Func == nil {
		return fmt.Errorf("cost function is nil")
	}
	if p.cost.Deriv == nil {
		return fmt.Errorf("cost derivative is nil")
	}
	return nil
}

//BackProp updates weight and bias based on the erreur
func (p *Perceptron) BackProp(x *mat.M64, err float64) {
	//cost := p.cost.Func(err)
	dcost := p.cost.Deriv(err) //derivative of the cost applied to the err
	dsig := p.f.Deriv(p.s)
	delta := p.alpha * dcost * dsig
	//fmt.Printf("cost: %f, dcost: %v, dsig: %v, delta: %f\n", cost, dcost, dsig, delta)

	p.b -= delta
	r, c := p.w.Dims()
	inputs := x.GetData()
	vals := make([]float64, x.Size())
	for i := range vals {
		vals[i] = delta * inputs[i]
	}
	p.w.Sub(mat.NewM64(r, c, vals))
	fmt.Printf("new w: %v, new b: %f\n", p.w.GetData(), p.b)
}

//NewPerceptron is a Peceptron constructor
func NewPerceptron(inSize int, learningRate float64, f, cost activation.F) (*Perceptron, error) {
	if inSize <= 0 {
		return nil, fmt.Errorf("input size is <=0")
	}
	if learningRate <= 0 || learningRate > 1 {
		return nil, fmt.Errorf("learning rate must be in ]0;1]")
	}
	p := &Perceptron{inSize: inSize, w: mat.NewM64(1, inSize, nil), f: f, cost: cost, alpha: learningRate, b: 0.0}
	err := p.Validate()
	return p, err
}
