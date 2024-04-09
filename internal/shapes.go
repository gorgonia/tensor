package internal

import (
	"golang.org/x/exp/constraints"
	"gorgonia.org/shapes"
)

// this file holds the internal shape utility functions

func ReduceShape(shape shapes.Shape, axis int) shapes.Shape {
	retVal := make(shapes.Shape, len(shape)-1)
	copy(retVal, shape[:axis])
	copy(retVal[axis:], shape[axis+1:])
	return retVal
}

func Prod[DT interface {
	constraints.Integer | constraints.Float
}](a []DT) (retVal DT) {
	retVal = DT(1)
	if len(a) == 0 {
		return
	}
	for _, v := range a {
		retVal *= v
	}
	return retVal
}

// if dims = 2 and axis -1 it returns the last dimension. In this case 1
func ResolveAxis(axis int, dims int) int {
	res := axis % dims
	if (res < 0 && dims > 0) || (res > 0 && dims < 0) {
		return res + dims
	}

	return res
}

func ElimInnermostOutermost(a, b shapes.Shape) shapes.Shape {
	a2 := a.Clone()
	return append(a2[:len(a)-1], b[1:]...)
}

func LargestShape(shps ...shapes.Shape) shapes.Shape {
	var maxShape shapes.Shape
	for _, s := range shps {
		if s.TotalSize() >= maxShape.TotalSize() && s.Dims() > maxShape.Dims() {
			maxShape = s
		}
	}
	return maxShape
}
