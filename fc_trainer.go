package nn

import (
	"fmt"
	"math"
	"math/rand"

	mat "github.com/klahssen/go-mat"
	"github.com/klahssen/nn/internal/activation"
)

const (
	minDropOut = 0.0
	maxDropOut = 0.9
)

//FCTrainer trains the inner Fully Connected feed forward neural network with training and validation datasets, and evaluates its performance with test dataset
type FCTrainer struct {
	n  *FC
	lr LrSource
	/*training   Dataset
	validation Dataset
	test       Dataset*/
	l       Logger
	cost    activation.F
	niter   uint
	maxiter uint
	tol     float64
}

//NewFCTrainer constructs a new Trainer for a Feed Forward Neural Net. It will stop if it reaches max number of iter or converges to the error tolerance
func NewFCTrainer(fc *FC, l Logger, lr LrSource, maxIter uint, tolerance float64, cost activation.F) (*FCTrainer, error) {
	t := &FCTrainer{n: fc, lr: lr, l: l, maxiter: maxIter, cost: cost, tol: math.Abs(tolerance)}
	err := t.validate()
	return t, err
}

//Validate checks if the trainers definition is OK
func (t *FCTrainer) validate() error {
	if t == nil {
		return fmt.Errorf("trainer is nil")
	}
	if t.n == nil {
		return fmt.Errorf("neural network is nil")
	}
	if t.lr == nil {
		return fmt.Errorf("learning rate source is nil")
	}
	//check network def
	var err error
	for i, lay := range t.n.layers {
		if err = lay.IsUsable(); err != nil {
			return fmt.Errorf("layer [%d]: %s", i, err.Error())
		}
	}
	if t.l == nil {
		//return fmt.Errorf("logger is nil")
		t.l = dftLogger
	}

	if t.maxiter < 0 {
		t.maxiter = 1
	}
	if t.cost.Func == nil {
		return fmt.Errorf("cost function is nil")
	}
	if t.cost.Deriv == nil {
		return fmt.Errorf("cost derivative is nil")
	}
	return nil
}

func (t *FCTrainer) withBackprop(r rand.Source, dataset Dataset, dropOutPeriod uint, dropOutRatio float64, batchSize uint) (float64, error) {
	avg := 0.0 //average cost
	if t == nil {
		return avg, fmt.Errorf("trainer is nil")
	}
	if t.l == nil {
		t.l = dftLogger
	}
	t.l.Printf("Check if trainable ")
	if err := t.validate(); err != nil {
		return avg, err
	}
	if r == nil {
		return avg, fmt.Errorf("random source r is nil")
	}
	if dropOutRatio <= minDropOut || dropOutRatio > maxDropOut {
		return avg, fmt.Errorf("dropout must be between %.1f and %.1f", minDropOut, maxDropOut)
	}
	if dataset == nil || dataset.Size() == 0 {
		return avg, fmt.Errorf("dataset is empty")
	}
	t.l.Printf("Start training ...")
	counter := uint(1)
	c := 0.0 //stores the cost for a point (average of cost of all outputs)
	var p *Datapoint
	var pred, dev, cost, gradCost *mat.M64
	var err error
	//t.l.Printf("Max iterations: %d\n", t.maxiter)
	for i := uint(1); i <= t.maxiter; i++ {
		//t.l.Printf("iteration: %d counter=%d\n", i, counter)
		dataset.Reset()
		ip := 0
		for {
			//process each datapoint
			p = dataset.Next()
			if p == nil {
				break
			}
			//t.l.Printf("point [%d]: %+v\n", ip, p)
			//compute prediction
			pred, err = t.n.FeedForward(p.Inp)
			if err != nil {
				return avg, fmt.Errorf("iteration %d: training point %d: %s", i, ip, err.Error())
			}
			//compute deviation
			dev, err = mat.Sub(pred, p.Exp)
			if err != nil {
				return avg, fmt.Errorf("iteration %d: training point %d: failed to compute deviation: %s", i, ip, err.Error())
			}
			cost, err = mat.MapElem(dev, t.cost.Func)
			if err != nil {
				return avg, fmt.Errorf("iteration %d: training point %d: failed to compute cost vector: %s", i, ip, err.Error())
			}
			c = 0.0
			for j := 0; j < cost.Size(); j++ {
				c += cost.AtInd(j)
			}
			c = c / float64(cost.Size())
			avg += c
			if counter == batchSize {
				//t.l.Printf("counter==batchSize: backprop!\n")
				gradCost, err = mat.MapElem(dev, t.cost.Deriv)
				if err != nil {
					return avg, fmt.Errorf("iteration %d: training point %d: failed to compute cost vector: %s", i, ip, err.Error())
				}
				counter = 0
				//compute mean error
				avg = avg / float64(batchSize)
				//backprop
				err = t.n.Backprop(t.lr.GetRate(), p.Inp, gradCost)
				if err != nil {
					return avg, fmt.Errorf("iteration %d: training point %d: failed to backpropagate: %s", i, ip, err.Error())
				}
				if avg <= t.tol {
					break
				}
			}
			counter++
			ip++
		}
	}

	t.l.Printf("Total Average Cost = %f", avg)
	return avg, nil
}

//testWith uses current definition of the Neural Network on a dataset and outputs the performance (average cost)
func (t *FCTrainer) testWith(data Dataset) (float64, error) {
	perf, c := 0.0, 0.0
	if data == nil || data.Size() == 0 {
		return perf, fmt.Errorf("no test data")
	}
	var dev, cost *mat.M64
	var p *Datapoint
	iters := 0
	data.Reset()
	t.l.Printf("Start Evaluation ...")
	for {
		p = data.Next()
		if p == nil {
			break
		}
		pred, err := t.n.FeedForward(p.Inp)
		if err != nil {
			return perf, fmt.Errorf("datapoint[%d]: %s", iters, err.Error())
		}
		//compute deviation
		dev, err = mat.Sub(pred, p.Exp)
		if err != nil {
			return perf, fmt.Errorf("training point %d: failed to compute deviation: %s", iters, err.Error())
		}
		cost, err = mat.MapElem(dev, t.cost.Func)
		if err != nil {
			return perf, fmt.Errorf("training point %d: failed to compute cost vector: %s", iters, err.Error())
		}
		c = 0.0
		for j := 0; j < cost.Size(); j++ {
			c += cost.AtInd(j)
		}
		c = c / float64(cost.Size())
		perf += c
		//update Weights and bias layer by layer (backpropagation) from -lr*gradient(d)
		iters++
	}
	perf = perf / float64(iters)
	t.l.Printf("Evaluation: Total Average Cost = %f", perf)
	return perf, nil
}

//TrainWithBackprop trains the inner network using back propagation, with an optional dropout (if period>0). updatePeriod sets how often backpropagation is applied and the period on which the cost is averaged. Deactivated neurons are selected randomly using the provided source
func (t *FCTrainer) TrainWithBackprop(r rand.Source, dropOutPeriod uint, dropOutRatio float64, batchSize uint, training, validation, test Dataset) (*FC, error) {
	if err := t.validate(); err != nil {
		return t.n, err
	}
	for i := range t.n.layers {
		t.n.layers[i].keepState = true
	}
	if training == nil || training.Size() == 0 {
		return t.n, fmt.Errorf("training set is empty")
	}
	if test == nil || test.Size() == 0 {
		return t.n, fmt.Errorf("test set is empty")
	}
	t.l.Printf("--- Training set ---\n")
	if _, err := t.withBackprop(r, training, dropOutPeriod, dropOutRatio, batchSize); err != nil {
		return t.n, err
	}
	//validation set
	if validation != nil && validation.Size() > 0 {
		t.l.Printf("--- Validation set ---\n")
		if _, err := t.withBackprop(r, validation, dropOutPeriod, dropOutRatio, 1); err != nil {
			return t.n, err
		}
	}
	t.l.Printf("--- Test set ---\n")
	_, err := t.testWith(test)
	if err != nil {
		return t.n, err
	}
	return t.n, nil
}
