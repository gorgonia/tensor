package axialiter

import "gorgonia.org/tensor/internal/errors"

type infoer interface {
	Info() *AP
}

// AxialIterator iterates based on a given axis
type AxialIterator struct {
	*AP
	axis int // the axis to iterate along

	// state
	axisSz    int // if an axis is of size N, then axisSz indicates the current num (0 - N).
	nextIndex int
	lastIndex int
	track     []int
	isReverse bool
	done      bool
	fixed     bool
}

// New creates and axial iterator that will iterate along the given axis. `fixedAxis` defines if the axisSz is fixed.
func New(ap *AP, axis, axisSz int, fixedAxis bool) *AxialIterator {
	return &AxialIterator{
		AP:     ap,
		track:  make([]int, len(ap.Shape())),
		axis:   axis,
		axisSz: axisSz,
		fixed:  fixedAxis,
	}
}

// Start returns the first index
func (it *AxialIterator) Start() (retVal int, err error) {
	it.Reset()

	// compute the nextIndex
	if it.fixed {
		it.track[it.axis] = it.axisSz
		it.nextIndex, err = Ltoi(it.Shape(), it.Strides(), it.track...)
	}

	return it.Next()
}

// Next returns the next index.
// Example: let's say we're iterating on a tensor with the following
//
//	shape: (2, 3, 4); axis: 1
//
// At the start, the coordinates are:
//
//	coordinates: (0, 0, 0)
//
// Next() will yield:
//
//	coordinates: (0, 0, 1)
//
// But when the coordinates are:
//
//	coordinates: (0, 0, 4)
//
// Next() will yield:
//
//	coordinates: (1, 0, 0).
//
// Note that axis 1 is frozen at 0.
func (it *AxialIterator) Next() (int, error) {
	if it.done {
		return -1, errors.NoOp{}
	}

	switch {
	case it.isReverse:
		return it.ndPrevious()
	default:
		return it.ndNext()
	}

}

func (it *AxialIterator) ndNext() (int, error) {
	shp := it.Shape()
	v := len(shp) - 1
	nextIndex := it.nextIndex
	it.lastIndex = nextIndex

	track := it.track[:v+1]       // force bounds check
	coord := shp[:v+1]            // force bounds check
	strides := it.Strides()[:v+1] // force bounds check
	sz := it.axisSz
	track[it.axis] = sz

	for i := v; i >= 0; i-- {
		if i == it.axis {
			if i == 0 {
				if it.fixed || track[it.axis] == coord[it.axis] || it.axisSz >= coord[it.axis] {
					track[it.axis] = 0
					it.done = true
					break
				}
				it.axisSz++
				track[it.axis] = it.axisSz
			}
			continue // we're iterating along an axis.
		}
		track[i]++
		shapeI := coord[i]
		strideI := strides[i]
		if track[i] == shapeI {
			track[i] = 0
			nextIndex -= (shapeI - 1) * strideI
			if i == 0 {
				it.axisSz++
				track[it.axis] = it.axisSz

				if it.fixed || track[it.axis] == coord[it.axis] || it.axisSz >= coord[it.axis] {
					track[it.axis] = 0
					it.done = true
					break
				}

				nextIndex = track[it.axis] * strides[it.axis]
			}

			continue
		}
		nextIndex += strideI
		break
	}
	it.nextIndex = nextIndex
	return it.lastIndex, nil
}

func (it *AxialIterator) ndPrevious() (int, error) {
	panic("Not yet implemented")
}

// NextValidity is like Next, but returns the validity of the value at the index as well.
func (it *AxialIterator) NextValidity() (int, bool, error) {
	i, err := it.Next()
	return i, true, err
}

// NextValid returns the next valid index, as well as a skip count.
func (it *AxialIterator) NextValid() (int, int, error) {
	if it.done {
		return -1, 1, errors.NoOp{}
	}

	switch {
	case it.isReverse:
		a, err := it.ndPrevious()
		return a, -1, err
	default:
		a, err := it.ndNext()
		return a, 1, err
	}
}

// NextInvalid returns the next invalid index, as well as a skip count.
func (it *AxialIterator) NextInvalid() (int, int, error) {
	panic("not implemented") // TODO: Implement
}

// Reset resets the iterator
func (it *AxialIterator) Reset() {
	it.nextIndex = 0
	for i := range it.track {
		it.track[i] = 0
	}
}

// SetReverse tells the iterator to iterate in reverse
func (it *AxialIterator) SetReverse() { it.isReverse = true }

// SetForward tells the iterator to iterate forwards
func (it *AxialIterator) SetForward() { it.isReverse = false }

// Coord returns the coordinates
func (it *AxialIterator) Coord() []int { return it.track }

// Done returns true when the iterator is done iterating.
func (it *AxialIterator) Done() bool { return it.done }
