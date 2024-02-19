package tensor

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal"
)

// CalcBroadcastShapes creates new shape for both A and B operands, on the assumption that autobroadcasting is used.
func CalcBroadcastShapes(a, b *AP) (newA, newB *AP) {
	aShp := a.Shape()
	bShp := b.Shape()

	aDim := aShp.Dims()
	bDim := bShp.Dims()

	// "clever" handling of different dims
	maxDim := internal.Max(aDim, bDim)
	newA = a
	if aDim != maxDim {
		newA = a.cloneWithNewShape(make(shapes.Shape, maxDim))
		copy(newA.shape[maxDim-aDim:], aShp)
		for i := 0; i < maxDim-aDim; i++ {
			newA.shape[i] = 1
		}
		newA.RecalcStrides()
	}
	newB = b
	if bDim != maxDim {
		newB = b.cloneWithNewShape(make(shapes.Shape, maxDim))
		copy(newB.shape[maxDim-bDim:], bShp)
		for i := 0; i < maxDim-bDim; i++ {
			newB.shape[i] = 1
		}
		newB.RecalcStrides()
	}

	return
}
