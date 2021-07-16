package tensor

import (
	"gorgonia.org/shapes"
)

var scalarShape = Shape{}

// ScalarShape represents a scalar. It has no dimensions, no sizes
func ScalarShape() Shape { return scalarShape }

// Shape represents a Shape. See the package shapes
type Shape = shapes.Shape

// CalcStrides calculates the default strides for a shape
func CalcStrides(s Shape) []int {
	if s.IsScalar() {
		return nil
	}

	retVal := BorrowInts(len(s))
	// if s.IsVector() {
	// 	retVal[0] = 1
	// 	retVal = retVal[:1]
	// 	return retVal
	// }

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

// CalcStridesWithMask is similar to CalcStrides, except that it has an argument, masks. It is used to mask out given dimensions
// during calculation of stride
func CalcStridesWithMask(s Shape, mask []bool) []int {
	if s.IsScalarEquiv() {
		return nil
	}

	retVal := BorrowInts(len(s))
	if s.IsVector() {
		retVal[0] = 1
		retVal = retVal[:1]
		return retVal
	}

	if len(mask) != s.Dims() {
		panic("mask length must be equal to number of shape dimensions")
	}
	acc := 1
	for i := len(s) - 1; i >= 0; i-- {
		if mask[i] {
			retVal[i] = acc
		} else {
			retVal[i] = 0
		}
		d := s[i]
		if d < 0 {
			panic("negative dimension size does not make sense")
		}
		if mask[i] {
			acc *= d
		}
	}

	return retVal
}

// CalcStridesColMajor is like CalcStrides, but assumes a col major layout
func CalcStridesColMajor(s Shape) []int {
	if s.IsScalarEquiv() {
		return nil
	}

	retVal := BorrowInts(len(s))
	if s.IsVector() {
		retVal[0] = 1
		retVal = retVal[:1]
		return retVal
	}

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
