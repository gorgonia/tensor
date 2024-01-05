package tensor

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

// CheckBroadcastable checks whether two given shapes are broadcastable with one another.
func CheckBroadcastable(aShp, bShp shapes.Shape) (err error) {
	if aShp.Eq(bShp) {
		return errors.NoOp{}
	}
	if aShp.Dims() != bShp.Dims() {
		return errors.Errorf(errors.DimMismatch, aShp.Dims(), bShp.Dims())
	}
	maxDim := aShp.Dims()

	// now, we check the shapes for broadcastability
	// if the shapes are not broadcastable, we return an error
	for i := maxDim - 1; i >= 0; i-- {
		bDim := aShp[i]
		aDim := bShp[i]
		if bDim != aDim && bDim != 1 && aDim != 1 {
			return errors.Errorf(errors.BroadcastError, aShp, bShp, i)
		}
	}
	return
}

// CalcBroadcastShapes creates new shape for both A and B operands, on the assumption that autobroadcasting is used.
func CalcBroadcastShapes(aShp, bShp shapes.Shape) (newShapeA, newShapeB shapes.Shape) {
	aDim := aShp.Dims()
	bDim := bShp.Dims()

	// "clever" handling of different dims
	maxDim := internal.Max(aDim, bDim)
	newShapeA = aShp
	if aDim != maxDim {
		newShapeA = make(shapes.Shape, maxDim)
		copy(newShapeA[maxDim-aDim:], aShp)
		for i := 0; i < maxDim-aDim; i++ {
			newShapeA[i] = 1
		}
	}
	newShapeB = bShp
	if bDim != maxDim {
		newShapeB = make(shapes.Shape, maxDim)
		copy(newShapeB[maxDim-bDim:], bShp)
		for i := 0; i < maxDim-bDim; i++ {
			newShapeB[i] = 1
		}
	}
	return
}
