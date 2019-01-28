package activation

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

//function types
const (
	FuncTypeSigmoid   = "sig"
	FuncTypeTanh      = "tanh"
	FuncTypeRelu      = "relu"
	FuncTypeLeakyRelu = "leaky_relu"
	FuncTypeElu       = "elu"
	FuncTypeIden      = "iden"
	FuncTypeCustom    = "custom"
)

var validFtypes = map[string]struct{}{FuncTypeSigmoid: {}, FuncTypeTanh: {}, FuncTypeRelu: {}, FuncTypeLeakyRelu: {}, FuncTypeElu: {}, FuncTypeIden: {}, FuncTypeCustom: {}}

//ValidateFType checks if valid function type
func ValidateFType(ftype string) error {
	if _, ok := validFtypes[ftype]; !ok {
		return fmt.Errorf("invalid activation function type '%s': expected one of [%s]", ftype, strings.Join(getValidFTypes(), ", "))
	}
	return nil
}

//getValidFuncTypes returns valid activation function types
func getValidFTypes() []string {
	valids := make([]string, len(validFtypes))
	i := 0
	for k := range validFtypes {
		valids[i] = k
		i++
	}
	sort.Strings(valids)
	return valids
}

//GetF generates an instance of F for a pair of activation function type and parameters
func GetF(ftype string, params []float64) (F, error) {
	if err := ValidateFType(ftype); err != nil {
		return F{}, err
	}
	switch ftype {
	case FuncTypeIden:
		return Iden(), nil
	case FuncTypeSigmoid:
		return Sigmoid(), nil
	case FuncTypeTanh:
		return Tanh(), nil
	case FuncTypeRelu:
		return Relu(), nil
	case FuncTypeLeakyRelu:
		nparams := 1
		if len(params) != nparams {
			return F{}, fmt.Errorf("expected %d parameter(s) for func '%s'", nparams, ftype)
		}
		return LeakyRelu(params[0]), nil
	case FuncTypeElu:
		nparams := 1
		if len(params) != nparams {
			return F{}, fmt.Errorf("expected %d parameter(s) for func '%s'", nparams, ftype)
		}
		return Elu(params[0]), nil
	default:
		return F{}, fmt.Errorf("invalid activation function type '%s': expected one of [%s]", ftype, strings.Join(getValidFTypes(), ", "))
	}
}

//F holds a function and its derivative
type F struct {
	Func  func(x float64) float64
	Deriv func(x float64) float64
}

//Sigmoid returns a sigmoid function with its derivative
func Sigmoid() F {
	return F{Func: sig, Deriv: derivSig}
}

//Sigmoid or logistic activation function
func sig(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//DerivSigmoid is Sigmoid's derivative
func derivSig(x float64) float64 {
	return sig(x) * (1 - sig(x))
}

//Tanh returns hyperbolic tangent and its derivative
func Tanh() F {
	return F{Func: tanh, Deriv: derivTanh}
}

//Tanh or hyperbolic tangent
func tanh(x float64) float64 {
	return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
}

//DerivTanh is Tanh's derivative
func derivTanh(x float64) float64 {
	return 1 - tanh(x)
}

//Elu returns an exponential linear unit with its derivative
func Elu(alpha float64) F {
	return F{Func: newElu(alpha), Deriv: newDerivElu(alpha)}
}

//NewElu returns a parametrized Exponential Linear Unit
func newElu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return x
		}
		return alpha * (math.Exp(x) - 1)
	}
}

//NewDerivElu returns the derivative of a parametrized Exponential Linear Unit
func newDerivElu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return alpha * math.Exp(x)
	}
}

//Relu returns a rectified linear unit and its derivative
func Relu() F {
	return F{Func: relu, Deriv: derivRelu}
}

//Relu rectified linear unit
func relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

//DerivRelu is Relu's derivative. Undefined for x=0
func derivRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

//LeakyRelu returns a leaky rectified linear unit and its derivative
func LeakyRelu(alpha float64) F {
	return F{Func: newLeakyRelu(alpha), Deriv: newDerivLeakyRelu(alpha)}
}

//NewLeakyRelu adds a slight slope for x<=0
func newLeakyRelu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return x
		}
		return alpha * x
	}
}

//NewDerivLeakyRelu is the derivative of a parametrized LeakyRelu. Undefined for x=0
func newDerivLeakyRelu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return alpha
	}
}

//Iden returns the identity function and its derivative
func Iden() F {
	return F{Func: iden, Deriv: derivIden}
}

//iden returns x
func iden(x float64) float64 {
	return x
}

//derivIden is iden's derivative
func derivIden(x float64) float64 {
	return 1
}

//Power returns x^n and its derivative
func Power(coef float64, n uint) F {
	return F{Func: newPower(coef, n), Deriv: newDerivPower(coef, n)}
}

func newPower(coef float64, n uint) func(x float64) float64 {
	return func(x float64) float64 {
		return coef * math.Pow(x, float64(n))
	}
}
func newDerivPower(coef float64, n uint) func(x float64) float64 {
	return func(x float64) float64 {
		return coef * float64(n) * math.Pow(x, float64(n-1))
	}
}

//Abs returnsthe absolute value function and its derivative
func Abs() F {
	return F{Func: math.Abs, Deriv: derivAbs}
}

func derivAbs(x float64) float64 {
	if x < 0 {
		return -1.0
	}
	return 1.0
}

//Softmax?
