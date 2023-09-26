package tensor

import (
	"gorgonia.org/tensor/internal/errors"
	gutils "gorgonia.org/tensor/internal/utils"
)

func CalcStridesColMajor(s Shape) []int {
	if s.IsScalar() {
		return nil
	}

	//retVal := BorrowInts(len(s))
	retVal := make([]int, len(s))
	// if s.IsVector() {
	// 	retVal[0] = 1
	// 	retVal = retVal[:1]
	// 	return retVal
	// }

	acc := 1
	for i := 0; i < len(s); i++ {
		retVal[i] = acc
		d := s[i]
		if d < 0 {
			panic("negative dimension size does not make sense")
		}
		acc *= d
	}
	return retVal
}

func CalcStrides(s Shape) []int {
	if s.IsScalar() {
		return nil
	}

	// retVal := BorrowInts(len(s))
	retVal := make([]int, len(s))

	acc := 1
	for i := len(s) - 1; i >= 0; i-- {
		retVal[i] = acc
		d := s[i]
		if d < 0 {
			panic("negative dimension size does not make sense")
		}
		acc *= d
	}
	return retVal
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
			err = errors.Errorf(errors.DimMismatch, len(shape), i)
			return
		}

		size := shape[i]

		if coord >= size {
			err = errors.Errorf(errors.IndexOOBAxis, i, coord, size)
			return
		}

		var stride int
		switch {
		case shape.IsVector() && len(strides) == 1:
			stride = strides[0]
		case i >= len(strides):
			err = errors.Errorf(errors.DimMismatch, len(strides), i)
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
		coord, i = gutils.Divmod(i, strides[d])

		if coord >= shape[d] {
			err = errors.Errorf(errors.IndexOOBAxis, d, coord, shape[d])
			// return
		}

		coords = append(coords, coord)
	}
	return
}

func allones(a []int) bool {
	for i := range a {
		if a[i] != 1 {
			return false
		}
	}
	return true
}

// CheckSlice checks a slice to see if it's sane
func CheckSlice(s SliceRange, size int) error {
	start := s.Start()
	end := s.End()
	step := s.Step()

	if start > end {
		return errors.Errorf(errors.InvalidSliceIndex, start, end)
	}

	if start < 0 {
		return errors.Errorf(errors.InvalidSliceIndex, start, 0)
	}

	if step == 0 && end-start > 1 {
		return errors.Errorf("Slice has 0 steps. Start is %d and end is %d", start, end)
	}

	if start >= size {
		return errors.Errorf("Start %d is greater than size %d", start, size)
	}

	return nil
}

// IsMonotonicInts returns true if the slice of ints is monotonically increasing. It also returns true for incr1 if every succession is a succession of 1
func IsMonotonicInts(axes []int) (monotonic bool, incr1 bool) {
	if len(axes) == 0 {
		return true, false
	}

	incr1 = true
	for i := 1; i < len(axes); i++ {
		if axes[i] < axes[i-1] {
			return false, axes[i] == axes[i-1]+1
		}
		if axes[i] != axes[i-1]+1 {
			incr1 = false
		}
	}
	return true, incr1
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
				err = errors.Errorf(errors.DimMismatch, len(x), len(pattern))
				return
			}
		} else {
			if len(x) != dims {
				err = errors.Errorf(errors.DimMismatch, len(x), len(pattern))
				return
			}
		}
	}

	// check that all the axes are < nDims
	// and that there are no axis repeated
	seen := make(map[int]struct{})
	for _, a := range pattern {
		if a >= dims {
			err = errors.Errorf(errors.InvalidAxis, a, dims)
			return
		}

		if _, ok := seen[a]; ok {
			err = errors.Errorf(errors.RepeatedAxis, a)
			return
		}

		seen[a] = struct{}{}
	}

	// no op really... we did the checks for no reason too. Maybe move this up?
	if monotonic, incr1 := IsMonotonicInts(pattern); monotonic && incr1 {
		err = errors.NoOp{}
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

// SliceDetails is a function that takes a slice and spits out its details. The whole reason for this is to handle the nil Slice, which is this: a[:]
func SliceDetails(s SliceRange, size int) (start, end, step int, err error) {
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
