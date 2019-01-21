package activation

import "math"

//Sigmoid or logistic activation function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

//DerivSigmoid is Sigmoid's derivative
func DerivSigmoid(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

//Tanh or hyperbolic tangent
func Tanh(x float64) float64 {
	return (math.Exp(x) - math.Exp(-x)) / (math.Exp(x) + math.Exp(-x))
}

//DerivTanh is Tanh's derivative
func DerivTanh(x float64) float64 {
	return 1 - Tanh(x)
}

//NewElu returns a parametrized Exponential Linear Unit
func NewElu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return x
		}
		return alpha * (math.Exp(x) - 1)
	}
}

//NewDerivElu returns the derivative of a parametrized Exponential Linear Unit
func NewDerivElu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return alpha * math.Exp(x)
	}
}

//Relu rectified linear unit
func Relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

//DerivRelu is Relu's derivative. Undefined for x=0
func DerivRelu(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

//NewLeakyRelu adds a slight slope for x<=0
func NewLeakyRelu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return x
		}
		return alpha * x
	}
}

//NewDerivLeakyRelu is the derivative of a parametrized LeakyRelu. Undefined for x=0
func NewDerivLeakyRelu(alpha float64) func(x float64) float64 {
	return func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return alpha
	}
}

//Softmax?
