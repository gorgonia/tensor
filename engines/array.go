package stdeng

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal"
)

func assign[DT any](dst, src []DT, dd, sd int, dstStrides, srcStrides []int, dstShape, srcShape shapes.Shape) error {
	ds := dstStrides[0]
	ss := srcStrides[sd-1]

	// when dd == 1, and the strides point in the same direction
	// we copy to a temporary if there is an overlap of data
	if ((dd == 1 && sd >= 1 && ds*ss < 0) || dd > 1) && internal.Overlaps(dst, src) {
		// create temp
		// copiedSrc = true
	}

	// broadcast src to dest for raw iteration
	tmpShape := make(shapes.Shape, sd)
	tmpStrides := make([]int, len(srcStrides))
	copy(tmpShape, srcShape)
	copy(tmpStrides, srcStrides)

	if sd > dd {
		tmpDim := sd
		for tmpDim > dd && tmpShape[0] == 1 {
			tmpDim--

			// this is better than tmpShape = tmpShape[1:]
			// because we are going to return these ints later
			copy(tmpShape, tmpShape[1:])
			copy(tmpStrides, tmpStrides[1:])
		}
	}

}
