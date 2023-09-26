package dense

import "gorgonia.org/shapes"

func reduceShape(shape shapes.Shape, axis int) shapes.Shape {
	retVal := make(shapes.Shape, len(shape)-1)
	copy(retVal, shape[:axis])
	copy(retVal[axis:], shape[axis+1:])
	return retVal
}

// if dims = 2 and axis -1 it returns the last dimension. In this case 1
func resolveAxis(axis int, dims int) int {
	res := axis % dims
	if (res < 0 && dims > 0) || (res > 0 && dims < 0) {
		return res + dims
	}

	return res
}

func elimInnermostOutermost(a, b shapes.Shape) shapes.Shape {
	a2 := a.Clone()
	return append(a2[:len(a)-1], b[1:]...)
}
