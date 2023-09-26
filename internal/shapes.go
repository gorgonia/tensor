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
