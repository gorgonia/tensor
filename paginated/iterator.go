package paginated

import "gorgonia.org/tensor"

// Iterator is an iterator that
// can be used to iterate over a paginated
// tensor.
type Iterator struct {
	*Tensor
	coord   []int
	reverse bool
}

// NewIterator will return an iterator for the provided
// paginated tensor.
func NewIterator(p *Tensor) tensor.Iterator {
	return &Iterator{
		Tensor: p,
		coord:  make([]int, p.Dims(), p.Dims()),
	}
}

// Start returns the first index
func (p *Iterator) Start() (int, error) {
	var start int
	if p.reverse {
		start = p.Tensor.globalIndex(p.largestCoord()...)
	}
	return start, nil
}

// Next returns the next index. Next is defined as the next value in the coordinates
// For example: let x be a (5,5) matrix that is row-major. Current index is for the coordinate (3,3).
// Next() returns the index of (3,4).
//
// If there is no underlying data store for (3,4) - say for example, the matrix is a sparse matrix, it return an error.
// If however, there is an underlying data store for (3,4), but it's not valid (for example, masked tensors), it will not return an error.
//
// Second example: let x be a (5,5) matrix that is col-major. Current index is for coordinate (3,3).
// Next() returns the index of (4,3).
func (p *Iterator) Next() (int, error) {
	if !p.reverse {
		p.coord = p.Tensor.nextCoord(p.coord)
	} else {
		p.coord = p.Tensor.previousCoord(p.coord)
	}

	return p.Tensor.globalIndex(p.coord...), nil
}

// NextValidity is like Next, but returns the validity of the value at the index as well.
// This will always return `true` and the same values as `iterator.Next()` because a paginated
// tensor is treated as a Gorgonia `*Dense` tensor.
func (p *Iterator) NextValidity() (int, bool, error) {
	next, _ := p.Next()
	return next, true, nil
}

// NextValid returns the next valid index, as well as a skip count.
// This function is the same as `NextValidity()` for iterators over
// paginated tensors.
func (p *Iterator) NextValid() (int, int, error) {
	next, _ := p.Next()
	return next, 0, nil
}

// NextInvalid always return an `ErrNotImplemented`.
func (p *Iterator) NextInvalid() (int, int, error) {
	return 0, 0, ErrNotImplemented
}

// Reset resets the iterator
func (p *Iterator) Reset() {
	if p.reverse {
		p.coord = p.largestCoord()
		return
	}

	p.coord = make([]int, p.Dims(), p.Dims())
}

// SetReverse tells the iterator to iterate in reverse
func (p *Iterator) SetReverse() {
	p.reverse = true
}

// SetForward tells the iterator to iterate forwards
func (p *Iterator) SetForward() {
	p.reverse = false
}

// Coord returns the coordinates of the current value
func (p *Iterator) Coord() []int {
	return p.coord
}

// Done returns true when the iterator is done iterating.
func (p *Iterator) Done() bool {
	for i, dim := range p.coord {
		if dim < (p.Tensor.Shape()[i] - 1) {
			return false
		}
	}
	return true
}

// Shape returns the shape of the multidimensional tensor it's iterating on.
func (p *Iterator) Shape() tensor.Shape {
	return p.dims
}
