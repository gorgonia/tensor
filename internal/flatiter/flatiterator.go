package flatiter

import (
	"github.com/chewxy/inigo/values/tensor"
	"github.com/chewxy/inigo/values/tensor/internal/errors"
)

/* FLAT ITERATOR */

// FlatIterator is an iterator that iterates over Tensors according to the data's layout.
// It utilizes the *AP of a Tensor to determine what the next index is.
// This data structure is similar to Numpy's flatiter, with some standard Go based restrictions of course
// (such as, not allowing negative indices)
type FlatIterator struct {
	*AP

	//state
	track      []int
	nextIndex  int
	lastIndex  int
	size       int
	done       bool
	veclikeDim int  // the dimension of a  vectorlike shape that is not a 1.
	reverse    bool // if true, iterator starts at end of array and runs backwards

	isScalar bool
	isVector bool

	outerFirst bool
}

// New creates a new FlatIterator.
func New(ap *AP) *FlatIterator {
	var dim int
	if ap.IsVectorLike() {
		for d, i := range ap.Shape() {
			if i != 1 {
				dim = d
				break
			}
		}
	}

	return &FlatIterator{
		AP:         ap,
		track:      make([]int, len(ap.Shape())),
		size:       ap.Shape().TotalSize(),
		veclikeDim: dim,

		isScalar: ap.IsScalar(),
		isVector: ap.IsVectorLike(),
	}
}

// SetReverse initializes iterator to run backwards
func (it *FlatIterator) SetReverse() {
	it.reverse = true
	it.Reset()
	return
}

// SetForward initializes iterator to run forwards
func (it *FlatIterator) SetForward() {
	it.reverse = false
	it.Reset()
	return
}

// Start begins iteration
func (it *FlatIterator) Start() (int, error) {
	it.Reset()
	return it.Next()
}

// Done checks whether iterators are done
func (it *FlatIterator) Done() bool {
	return it.done
}

// Next returns the index of the current coordinate.
func (it *FlatIterator) Next() (int, error) {
	if it.done {
		return -1, errors.NoOp{}
	}

	switch {
	case it.isScalar:
		it.done = true
		return 0, nil
	case it.isVector:
		if it.reverse {
			return it.singlePrevious()
		}
		return it.singleNext()
	default:
		if it.reverse {
			return it.ndPrevious()
		}
		if it.outerFirst {
			return it.colMajorNDNext()
		}
		return it.ndNext()
	}
}

// NextValidity returns the index of the current coordinate, and whether or not it's valid. Identical to Next()
func (it *FlatIterator) NextValidity() (int, bool, error) {
	i, err := it.Next()
	return i, true, err
}

// NextValid returns the index of the current coordinate. Identical to Next for FlatIterator
// Also returns the number of increments to get to next element ( 1,  or -1 in reverse case). This is to maintain
// consistency with the masked iterator, for which the step between valid elements can be more than 1
func (it *FlatIterator) NextValid() (int, int, error) {
	if it.done {
		return -1, 1, errors.NoOp{}
	}
	switch {
	case it.isScalar:
		it.done = true
		return 0, 0, nil
	case it.isVector:
		if it.reverse {
			a, err := it.singlePrevious()
			return a, -1, err
		}
		a, err := it.singleNext()
		return a, 1, err
	default:
		if it.reverse {
			a, err := it.ndPrevious()
			return a, -1, err
		}

		if it.outerFirst {
			a, err := it.colMajorNDNext()
			return a, 1, err
		}
		a, err := it.ndNext()
		return a, 1, err
	}
}

// NextInvalid returns the index of the current coordinate. Identical to Next for FlatIterator
// also returns the number of increments to get to next invalid element (1 or -1 in reverse case).
// Like NextValid, this method's purpose is to maintain consistency with the masked iterator,
// for which the step between invalid elements can be anywhere from 0 to the  tensor's length
func (it *FlatIterator) NextInvalid() (int, int, error) {
	if it.reverse {
		return -1, -it.lastIndex, errors.NoOp{}
	}
	return -1, it.Size() - it.lastIndex, errors.NoOp{}
}

func (it *FlatIterator) singleNext() (int, error) {
	it.lastIndex = it.nextIndex
	it.nextIndex++

	var tracked int
	it.track[it.veclikeDim]++
	tracked = it.track[it.veclikeDim]

	if tracked >= it.size {
		it.done = true
	}

	return it.lastIndex, nil
}

func (it *FlatIterator) singlePrevious() (int, error) {
	it.lastIndex = it.nextIndex
	it.nextIndex--

	var tracked int
	it.track[it.veclikeDim]--
	tracked = it.track[it.veclikeDim]

	if tracked < 0 {
		it.done = true
	}
	return it.lastIndex, nil
}

func (it *FlatIterator) ndNext() (int, error) {
	// the reason for this weird looking bits of code is because the SSA compiler doesn't
	// know how to optimize for this bit of code, not keeping things in registers correctly
	// @stuartcarnie optimized this iout to great effect

	v := len(it.Shape()) - 1
	nextIndex := it.nextIndex
	it.lastIndex = nextIndex

	// the following 3 lines causes the compiler to perform bounds check here,
	// instead of being done in the loop
	coord := it.Shape()[:v+1]
	track := it.track[:v+1]
	strides := it.Strides()[:v+1]
	for i := v; i >= 0; i-- {
		track[i]++
		shapeI := coord[i]
		strideI := strides[i]

		if track[i] == shapeI {
			if i == 0 {
				it.done = true
			}
			track[i] = 0
			nextIndex -= (shapeI - 1) * strideI
			continue
		}
		nextIndex += strideI
		break
	}
	it.nextIndex = nextIndex
	return it.lastIndex, nil
}

func (it *FlatIterator) colMajorNDNext() (int, error) {
	// the reason for this weird looking bits of code is because the SSA compiler doesn't
	// know how to optimize for this bit of code, not keeping things in registers correctly
	// @stuartcarnie optimized this iout to great effect

	v := len(it.Shape()) - 1
	nextIndex := it.nextIndex
	it.lastIndex = nextIndex

	// the following 3 lines causes the compiler to perform bounds check here,
	// instead of being done in the loop
	coord := it.Shape()[:v+1]
	track := it.track[:v+1]
	strides := it.Strides()[:v+1]
	for i := 0; i <= v; i++ {
		track[i]++
		shapeI := coord[i]
		strideI := strides[i]

		if track[i] == shapeI {
			if i == v {
				it.done = true
			}
			track[i] = 0

			nextIndex -= (shapeI - 1) * strideI
			continue
		}
		nextIndex += strideI
		break
	}
	it.nextIndex = nextIndex
	return it.lastIndex, nil

}

func (it *FlatIterator) ndPrevious() (int, error) {
	it.lastIndex = it.nextIndex
	for i := len(it.Shape()) - 1; i >= 0; i-- {
		it.track[i]--
		if it.track[i] < 0 {
			if i == 0 {
				it.done = true
			}
			it.track[i] = it.Shape()[i] - 1
			it.nextIndex += (it.Shape()[i] - 1) * it.Strides()[i]
			continue
		}
		it.nextIndex -= it.Strides()[i]
		break
	}
	return it.lastIndex, nil
}

// TODO v0.9.0
func (it *FlatIterator) colMajorNDPrevious() (int, error) {
	return 0, nil
}

// Coord returns the next coordinate.
// When Next() is called, the coordinates are updated AFTER the Next() returned.
// See example for more details.
//
// The returned coordinates is mutable. Changing any values in the return value will
// change the state of the iterator
func (it *FlatIterator) Coord() []int { return it.track }

// Slice is a convenience function that augments
func (it *FlatIterator) Slice(sli SliceRange) (retVal []int, err error) {
	var next int
	var nexts []int
	for next, err = it.Next(); err == nil; next, err = it.Next() {
		nexts = append(nexts, next)
	}
	if _, ok := err.(NoOpError); err != nil && !ok {
		return
	}

	if sli == nil {
		retVal = nexts
		return
	}

	start := sli.Start()
	end := sli.End()
	step := sli.Step()

	// sanity checks
	if err = tensor.CheckSlice(sli, len(nexts)); err != nil {
		return
	}

	if step < 0 {
		// reverse the nexts
		for i := len(nexts)/2 - 1; i >= 0; i-- {
			j := len(nexts) - 1 - i
			nexts[i], nexts[j] = nexts[j], nexts[i]
		}
		step = -step
	}

	// cleanup before loop
	if end > len(nexts) {
		end = len(nexts)
	}
	// nexts = nexts[:end]

	for i := start; i < end; i += step {
		retVal = append(retVal, nexts[i])
	}

	err = nil
	return
}

// Reset resets the iterator state.
func (it *FlatIterator) Reset() {
	it.done = false
	if it.reverse {
		for i := range it.track {
			it.track[i] = it.Shape()[i] - 1
		}

		switch {
		case it.IsScalar():
			it.nextIndex = 0
		case it.isVector:
			it.nextIndex = (it.Shape()[0] - 1) * it.Strides()[0]
		// case it.IsRowVec():
		// 	it.nextIndex = (it.Shape()[1] - 1) * it.Strides()[1]
		// case it.IsColVec():
		// 	it.nextIndex = (it.Shape()[0] - 1) * it.Strides()[0]
		default:
			it.nextIndex = 0
			for i := range it.track {
				it.nextIndex += (it.Shape()[i] - 1) * it.Strides()[i]
			}
		}
	} else {
		it.nextIndex = 0
		for i := range it.track {
			it.track[i] = 0
		}
	}
}

// Chan returns a channel of ints. This is useful for iterating multiple Tensors at the same time.
func (it *FlatIterator) Chan() (retVal chan int) {
	retVal = make(chan int)

	go func() {
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			retVal <- next
		}
		close(retVal)
	}()

	return
}
