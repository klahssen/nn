package nn

import (
	"fmt"
	"math"
	"math/rand"

	mat "github.com/klahssen/go-mat"
)

//FFNTrainer trains the inner FeedFwd network with training and validation datasets, and evaluates its performance with test dataset
type FFNTrainer struct {
	n          *FFN
	training   Dataset
	validation Dataset
	test       Dataset
	l          Logger
	niter      uint
	maxiter    uint
	tol        float64
}

//NewFFNTrainer constructs a new Trainer for a Feed Forward Neural Net. It will stop if it reaches max number of iter or converges to the error tolerance
func NewFFNTrainer(ff *FFN, l Logger, training, validation, test Dataset, maxIter uint, tolerance float64) (*FFNTrainer, error) {
	t := &FFNTrainer{n: ff, training: training, validation: validation, test: test, l: l, maxiter: maxIter, tol: math.Abs(tolerance)}
	err := t.Validate()
	return t, err
}

//Validate checks if the trainers definition is OK
func (t *FFNTrainer) Validate() error {
	if t == nil {
		return fmt.Errorf("trainer is nil")
	}
	if t.n == nil {
		return fmt.Errorf("neural network is nil")
	}
	//check network def
	var err error
	for i, lay := range t.n.layers {
		if err = lay.IsUsable(); err != nil {
			return fmt.Errorf("layer [%d]: %s", i, err.Error())
		}
	}
	if t.l == nil {
		return fmt.Errorf("logger is nil")
	}
	if t.training == nil || t.training.Size() == 0 {
		return fmt.Errorf("training set is empty")
	}
	if t.test == nil || t.test.Size() == 0 {
		return fmt.Errorf("test set is empty")
	}
	if t.maxiter < 20 {
		t.maxiter = 20
	}
	return nil
}

func (t *FFNTrainer) trainWithBackprop(r rand.Source, dropOutPeriod uint, dropOutRatio float64, updatePeriod uint, costFunc func(x float64) float64) (float64, error) {
	cost := 0.0
	t.l.Printf("Check if trainable ")
	if r == nil {
		return cost, fmt.Errorf("random source r is nil")
	}
	if dropOutRatio <= 0 || dropOutRatio > 0.9 {
		return cost, fmt.Errorf("dropout must be between 0 and 0.9")
	}
	//training set
	//train := true

	costi := 0.0 //average cost
	var data *Datapoint
	t.training.Reset()
	t.l.Printf("Start training ...")
	iters := 0
	for {
		data = t.training.Next()
		if data == nil {
			break
		}
		pred, err := t.n.Feed(data.Inp)
		if err != nil {
			return cost, fmt.Errorf("failed during training: datapoint[%d]: %s", iters, err.Error())
		}
		//compute loss
		d, err := mat.Sub(data.Exp, pred)
		if err != nil {
			return cost, fmt.Errorf("failed during training: datapoint[%d]: deviation: %s", iters, err.Error())
		}
		err = d.MapElem(costFunc)
		if err != nil {
			return cost, fmt.Errorf("failed during training: datapoint[%d]: cost: %s", iters, err.Error())
		}

		costi = 0.0
		for i := 0; i < d.Size(); i++ {
			costi += d.AtInd(i)
		}
		cost += costi /// float64(nout)
		//update Weights and bias layer by layer (backpropagation) from -lr*gradient(d)
		iters++
	}
	cost = cost / float64(iters)
	t.l.Printf("Training: Total Average Cost = %f", cost)
	return cost, nil
}

//runTest uses current definition of the Neural Network on a dataset and outputs the performance (average cost)
func (t *FFNTrainer) runTest(costFunc func(x float64) float64) (float64, error) {
	if t.test == nil || t.test.Size() == 0 {
		return 0, fmt.Errorf("no test data")
	}
	if costFunc == nil {
		return 0.0, fmt.Errorf("cost function is nil")
	}
	cost := 0.0
	costi := 0.0 //average cost
	var data *Datapoint
	t.test.Reset()
	t.l.Printf("Start Evaluation ...")
	iters := 0
	for {
		data = t.test.Next()
		if data == nil {
			break
		}
		pred, err := t.n.Feed(data.Inp)
		if err != nil {
			return cost, fmt.Errorf("failed during training: datapoint[%d]: %s", iters, err.Error())
		}
		//compute loss
		d, err := mat.Sub(data.Exp, pred)
		if err != nil {
			return cost, fmt.Errorf("failed during training: datapoint[%d]: deviation: %s", iters, err.Error())
		}
		err = d.MapElem(costFunc)
		if err != nil {
			return cost, fmt.Errorf("failed during training: datapoint[%d]: cost: %s", iters, err.Error())
		}

		costi = 0.0
		for i := 0; i < d.Size(); i++ {
			costi += d.AtInd(i)
		}
		cost += costi /// float64(nout)
		//update Weights and bias layer by layer (backpropagation) from -lr*gradient(d)
		iters++
	}
	cost = cost / float64(iters)
	t.l.Printf("Evaluation: Total Average Cost = %f", cost)
	return cost, nil
}

//WithBackprop trains the inner network using back propagation, with an optional dropout (if period>0). updatePeriod sets how often backpropagation is applied and the period on which the cost is averaged. Deactivated neurons are selected randomly using the provided source
func (t *FFNTrainer) WithBackprop(r rand.Source, dropOutPeriod uint, dropOutRatio float64, updatePeriod uint, costFunc func(x float64) float64) (*FFN, error) {
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if _, err := t.trainWithBackprop(r, dropOutPeriod, dropOutRatio, updatePeriod, costFunc); err != nil {
		return t.n, err
	}
	//validation set
	if t.validation != nil && t.validation.Size() > 0 {
		//	return t.n, err
	}
	//test set
	_, err := t.runTest(costFunc)
	if err != nil {
		return t.n, err
	}
	return t.n, nil
}
