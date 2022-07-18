package paginated

import "gorgonia.org/tensor"

// RequiresIterator returns false for a paginated tensor
func (p *Tensor) RequiresIterator() bool {
	return false
}

// Iterator will return a `*Iterator` object
// that can be used to iterate over the tensor.
func (p *Tensor) Iterator() tensor.Iterator {
	return NewIterator(p)
}

// DataOrder returns the ordering of the data of the tensor.
func (p *Tensor) DataOrder() tensor.DataOrder {
	return p.dataOrder
}
