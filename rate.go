package nn

//LrSource is an interface for a learning rate source
type LrSource interface {
	GetRate() float64
}

//Lr is a constant learning rate source
type Lr struct {
	val float64
}

//GetRate to implement LrSource
func (l *Lr) GetRate() float64 {
	return l.val
}

//NewLr returns an instance of a constant learning rate source
func NewLr(rate float64) *Lr {
	return &Lr{val: rate}
}
