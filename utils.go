package tensor

import (
	"context"

	"github.com/pkg/errors"
)

const AllAxes int = -1

// MinInt returns the lowest between two ints. If both are the  same it returns the first
func MinInt(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// MaxInt returns the highest between two ints. If both are the same, it  returns the first
func MaxInt(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

// MaxInts returns the max of a slice of ints.
func MaxInts(is ...int) (retVal int) {
	for _, i := range is {
		if i > retVal {
			retVal = i
		}
	}
	return
}

// SumInts sums a slice of ints
func SumInts(a []int) (retVal int) {
	for _, v := range a {
		retVal += v
	}
	return
}

// ProdInts returns the internal product of an int slice
func ProdInts(a []int) (retVal int) {
	retVal = 1
	if len(a) == 0 {
		return
	}
	for _, v := range a {
		retVal *= v
	}
	return
}

// IsMonotonicInts returns true if the slice of ints is monotonically increasing. It also returns true for incr1 if every succession is a succession of 1
func IsMonotonicInts(a []int) (monotonic bool, incr1 bool) {
	var prev int
	incr1 = true
	for i, v := range a {
		if i == 0 {
			prev = v
			continue
		}

		if v < prev {
			return false, false
		}
		if v != prev+1 {
			incr1 = false
		}
		prev = v
	}
	monotonic = true
	return
}

// Ltoi is Location to Index. Provide a shape, a strides, and a list of integers as coordinates, and returns the index at which the element is.
func Ltoi(shape Shape, strides []int, coords ...int) (at int, err error) {
	if shape.IsScalarEquiv() {
		for _, v := range coords {
			if v != 0 {
				return -1, errors.Errorf("Scalar shape only allows 0 as an index")
			}
		}
		return 0, nil
	}
	for i, coord := range coords {
		if i >= len(shape) {
			err = errors.Errorf(dimMismatch, len(shape), i)
			return
		}

		size := shape[i]

		if coord >= size {
			err = errors.Errorf(indexOOBAxis, i, coord, size)
			return
		}

		var stride int
		switch {
		case shape.IsVector() && len(strides) == 1:
			stride = strides[0]
		case i >= len(strides):
			err = errors.Errorf(dimMismatch, len(strides), i)
			return
		default:
			stride = strides[i]
		}

		at += stride * coord
	}
	return at, nil
}

// Itol is Index to Location.
func Itol(i int, shape Shape, strides []int) (coords []int, err error) {
	dims := len(strides)

	for d := 0; d < dims; d++ {
		var coord int
		coord, i = divmod(i, strides[d])

		if coord >= shape[d] {
			err = errors.Errorf(indexOOBAxis, d, coord, shape[d])
			// return
		}

		coords = append(coords, coord)
	}
	return
}

func UnsafePermute(pattern []int, xs ...[]int) (err error) {
	if len(xs) == 0 {
		err = errors.New("Permute requres something to permute")
		return
	}

	dims := -1
	patLen := len(pattern)
	for _, x := range xs {
		if dims == -1 {
			dims = len(x)
			if patLen != dims {
				err = errors.Errorf(dimMismatch, len(x), len(pattern))
				return
			}
		} else {
			if len(x) != dims {
				err = errors.Errorf(dimMismatch, len(x), len(pattern))
				return
			}
		}
	}

	// check that all the axes are < nDims
	// and that there are no axis repeated
	seen := make(map[int]struct{})
	for _, a := range pattern {
		if a >= dims {
			err = errors.Errorf(invalidAxis, a, dims)
			return
		}

		if _, ok := seen[a]; ok {
			err = errors.Errorf(repeatedAxis, a)
			return
		}

		seen[a] = struct{}{}
	}

	// no op really... we did the checks for no reason too. Maybe move this up?
	if monotonic, incr1 := IsMonotonicInts(pattern); monotonic && incr1 {
		err = noopError{}
		return
	}

	switch dims {
	case 0, 1:
	case 2:
		for _, x := range xs {
			x[0], x[1] = x[1], x[0]
		}
	default:
		for i := 0; i < dims; i++ {
			to := pattern[i]
			for to < i {
				to = pattern[to]
			}
			for _, x := range xs {
				x[i], x[to] = x[to], x[i]
			}
		}
	}
	return nil
}

// CheckSlice checks a slice to see if it's sane
func CheckSlice(s Slice, size int) error {
	start := s.Start()
	end := s.End()
	step := s.Step()

	if start > end {
		return errors.Errorf(invalidSliceIndex, start, end)
	}

	if start < 0 {
		return errors.Errorf(invalidSliceIndex, start, 0)
	}

	if step == 0 && end-start > 1 {
		return errors.Errorf("Slice has 0 steps. Start is %d and end is %d", start, end)
	}

	if start >= size {
		return errors.Errorf("Start %d is greater than size %d", start, size)
	}

	return nil
}

// SliceDetails is a function that takes a slice and spits out its details. The whole reason for this is to handle the nil Slice, which is this: a[:]
func SliceDetails(s Slice, size int) (start, end, step int, err error) {
	if s == nil {
		start = 0
		end = size
		step = 1
	} else {
		if err = CheckSlice(s, size); err != nil {
			return
		}

		start = s.Start()
		end = s.End()
		step = s.Step()

		if end > size {
			end = size
		}
	}
	return
}

// checkFixShape checks the shape and reshapes it to be correct if the size fits but the shape doesn't.
func checkFixShape(reuse Tensor, s Shape) (err error) {
	throw := BorrowInts(len(s))
	copy(throw, s)

	d, ok := reuse.(DenseTensor)
	if !ok {
		if err = reuse.Reshape(throw...); err != nil {
			return errors.Wrapf(err, reuseReshapeErr, s, reuse.DataSize())
		}
		return nil
	}

	if err = d.reshape(throw...); err != nil {
		err = errors.Wrapf(err, reuseReshapeErr, s, d.DataSize())
		return
	}

	// clean up any funny things that may be in the reuse
	if oldAP := d.oldAP(); !oldAP.IsZero() {
		oldAP.zero()
	}

	if axes := d.transposeAxes(); axes != nil {
		ReturnInts(axes)
	}

	if viewOf := d.parentTensor(); viewOf != nil {
		d.setParentTensor(nil)
	}
	return nil
}

// memsetBools sets boolean slice to value.
// Reference http://stackoverflow.com/questions/30614165/is-there-analog-of-memset-in-go
func memsetBools(a []bool, v bool) {
	if len(a) == 0 {
		return
	}
	a[0] = v
	for bp := 1; bp < len(a); bp *= 2 {
		copy(a[bp:], a[:bp])
	}
}

// allones checks that a slice of ints are all 1.
func allones(a []int) bool {
	for i := range a {
		if a[i] != 1 {
			return false
		}
	}
	return true
}

// ctxFromEngine gets a context from an engine if it's a contexter. Otherwise it returns a context.Background()
func ctxFromEngine(e Engine) context.Context {
	if c, ok := e.(contexter); ok {
		return c.Context()
	}
	return context.Background()
}

func getFloat64s(a Tensor) []float64 {
	if um, ok := a.(unsafeMem); ok {
		return um.Float64s()
	}
	return a.Data().([]float64)
}

func getFloat32s(a Tensor) []float32 {
	if um, ok := a.(unsafeMem); ok {
		return um.Float32s()
	}
	return a.Data().([]float32)
}

func getInts(a Tensor) []int {
	if um, ok := a.(unsafeMem); ok {
		return um.Ints()
	}
	return a.Data().([]int)

}

/* FOR ILLUSTRATIVE PURPOSES */

// Permute permutates a pattern according to xs. This function exists for illustrative purposes (i.e. the dumb, unoptimized version)
//
// In reality, the UnsafePermute function is used.
/*
func Permute(pattern []int, xs ...[]int) (retVal [][]int, err error) {
	if len(xs) == 0 {
		err = errors.New("Permute requires something to permute")
		return
	}

	dims := -1
	patLen := len(pattern)
	for _, x := range xs {
		if dims == -1 {
			dims = len(x)
			if patLen != dims {
				err = errors.Errorf(dimMismatch, len(x), len(pattern))
				return
			}
		} else {
			if len(x) != dims {
				err = errors.Errorf(dimMismatch, len(x), len(pattern))
				return
			}
		}
	}

	// check that all the axes are < nDims
	// and that there are no axis repeated
	seen := make(map[int]struct{})
	for _, a := range pattern {
		if a >= dims {
			err = errors.Errorf(invalidAxis, a, dims)
			return
		}

		if _, ok := seen[a]; ok {
			err = errors.Errorf(repeatedAxis, a)
			return
		}

		seen[a] = struct{}{}
	}

	// no op really... we did the checks for no reason too. Maybe move this up?
	if monotonic, incr1 := IsMonotonicInts(pattern); monotonic && incr1 {
		retVal = xs
		err = noopError{}
		return
	}

	switch dims {
	case 0, 1:
		retVal = xs
	case 2:
		for _, x := range xs {
			rv := []int{x[1], x[0]}
			retVal = append(retVal, rv)
		}
	default:
		retVal = make([][]int, len(xs))
		for i := range retVal {
			retVal[i] = make([]int, dims)
		}

		for i, v := range pattern {
			for j, x := range xs {
				retVal[j][i] = x[v]
			}
		}
	}
	return
}
*/
