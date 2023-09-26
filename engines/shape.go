package stdeng

import (
	"github.com/chewxy/inigo/values/tensor/internal"
	"gorgonia.org/shapes"
)

// asMat returns a matrix shape from the given shape and axis. The given axis is which dim it will stop in.
//
//	asMat((5), 0, true) = (1, 5)
//	asMat((5), 1, true) = (5, 1)
//	asMat((3,4,5), 0, true) = (1, 60)
//	asMat((3,4,5), 1, true) = (3, 20)
//	asMat((3,4,5), 2, true) = (12, 5)
//	asMat((3,4,5), 0, false) = (1, 20)
//	asMat((3,4,5), 1, false) = (3, 5)
//	asMat((3,4,5), 2, false) = (12, 1)
func asMat(a shapes.Shape, axis int, inclusive bool) (retVal shapes.Shape) {
	// no need to do a check because asMat will only ever be used by internal functions.

	retVal = make(shapes.Shape, 2)
	switch {
	case a.Dims() == 1 && axis == 0:
		retVal[0] = 1
		retVal[1] = a[0]
		return
	case a.Dims() == 1 && axis == 1:
		retVal[0] = a[0]
		retVal[1] = 1
		return
	}
	// outer
	retVal[0] = internal.Prod(a[:axis])
	aplus := axis
	if !inclusive {
		aplus++
	}
	// inner
	retVal[1] = internal.Prod(a[aplus:])
	return
}
