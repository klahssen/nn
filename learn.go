package nn

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	mat "github.com/klahssen/go-mat"
)

var dftLogger = &log.Logger{}

//Logger interface, injected in the
type Logger interface {
	Printf(format string, v ...interface{})
}

//Datapoint holds input data and expected output
type Datapoint struct {
	Inp *mat.M64 //[]float64
	Exp *mat.M64 //[]float64
}

//Dataset represents anything able to provide Datapoints, one at a time
type Dataset interface {
	Next() *Datapoint
	Size() int
	Left() int
	Reset()
}

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

//selectDrops provides a random selection of neurons to be deactivated
func selectDrops(r rand.Source, dropSize, fleetSize int) map[int]struct{} {
	if fleetSize <= 0 {
		return nil
	}
	if dropSize < 0 {
		dropSize *= -1
	}
	if dropSize > fleetSize {
		dropSize = fleetSize
	}
	if dropSize == 0 {
		return nil
	}
	m := map[int]struct{}{}
	counter := 0
	pick := 0
	for counter < dropSize {
		pick = int(r.Int63() % int64(fleetSize))
		if _, ok := m[pick]; !ok {
			m[pick] = struct{}{}
			counter++
		}
	}
	return m
}

//WithBackprop trains the inner network using back propagation, with an optional dropout (if period>0). Deactivated neurons are selected randomly using the provided source
func (t *FFNTrainer) WithBackprop(r rand.Source, dropOutPeriod uint, dropOutRatio float64, cost func(x float64) float64) (*FFN, error) {
	t.l.Printf("Check if trainable ")
	if err := t.Validate(); err != nil {
		return nil, err
	}
	if r == nil {
		return nil, fmt.Errorf("random source r is nil")
	}
	if dropOutRatio <= 0 || dropOutRatio > 0.9 {
		return nil, fmt.Errorf("dropout must be between 0 and 0.9")
	}
	//training set
	//train := true
	loss := 0.0
	avge := 0.0
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
			return t.n, fmt.Errorf("failed during training: datapoint[%d]: %s", iters, err.Error())
		}
		//compute loss
		d, err := mat.Sub(data.Exp, pred)
		if err != nil {
			return t.n, fmt.Errorf("failed during training: datapoint[%d]: deviation: %s", iters, err.Error())
		}
		err = d.MapElem(cost)
		if err != nil {
			return t.n, fmt.Errorf("failed during training: datapoint[%d]: cost: %s", iters, err.Error())
		}
		for i := 0; i < d.Size(); i++ {
			avge += d.AtInd(i)
		}
		loss += avge
		iters++
	}
	loss = loss / float64(iters)
	t.l.Printf("Training: Total Average Loss = %f", loss)
	iters = 0
	//validation set
	if t.validation != nil && t.validation.Size() > 0 {
		loss = 0.0
		var data *Datapoint
		t.training.Reset()
		t.l.Printf("Start validation ...")
		for {
			data = t.training.Next()
			if data == nil {
				break
			}

		}
		t.l.Printf("Validation: Total Average Loss = %f", loss)
	}
	//test set
	loss = 0.0
	t.l.Printf("Evaluate the model with test set ...")
	//print average error on the test set
	t.l.Printf("Evaluation: Total Average Loss = %f", loss)
	return t.n, nil
}
