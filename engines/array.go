package stdeng

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/flatiter"
)

// copyIter copies an array to another array using iterators
func copyIter[DT any](dst, src []DT, diter, siter internal.Iterator) int {
	var i, j, count int
	var err error
	for {
		if i, err = diter.Next(); err != nil {
			if err = internal.HandleNoOp(err); err != nil {
				panic(err)
			}
			break
		}
		if j, err = siter.Next(); err != nil {
			if err = internal.HandleNoOp(err); err != nil {
				panic(err)
			}
			break
		}
		dst[i] = src[j]
		count++
	}
	return count
}

// broadcastStrides handles broadcasting from different shapes.
//
// Deprecated: this function will be unexported
func broadcastStrides(destShape, srcShape shapes.Shape, destStrides, srcStrides []int) (retVal []int, err error) {
	dims := len(destShape)
	start := dims - len(srcShape)

	if destShape.IsVector() && srcShape.IsVector() {
		return []int{srcStrides[0]}, nil
	}

	if start < 0 {
		//error
		err = errors.Errorf(errors.DimMismatch, dims, len(srcShape))
		return
	}

	retVal = make([]int, len(destStrides))
	for i := dims - 1; i >= start; i-- {
		s := srcShape[i-start]
		switch {
		case s == 1:
			retVal[i] = 0
		case s != destShape[i]:
			// error
			err = errors.Errorf("Cannot broadcast from %v to %v", srcShape, destShape)
			return
		default:
			retVal[i] = srcStrides[i-start]
		}
	}
	for i := 0; i < start; i++ {
		retVal[i] = 0
	}
	return
}

func assign[DT any, T tensor.Basic[DT]](dst, src T) (err error) {
	dd := dst.Dims()
	sd := src.Dims()

	dstShape := dst.Shape()
	srcShape := src.Shape()
	dstStrides := dst.Strides()
	srcStrides := src.Strides()

	ds := dstStrides[0]
	ss := srcStrides[sd-1]

	// when dd == 1, and the strides point in the same direction
	// we copy to a temporary if there is an overlap of data
	if ((dd == 1 && sd >= 1 && ds*ss < 0) || dd > 1) && internal.Overlaps(dst.Data(), src.Data()) {
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

	var newStrides []int
	if newStrides, err = broadcastStrides(dstShape, tmpShape, dstStrides, tmpStrides); err != nil {
		err = errors.Wrapf(err, "BroadcastStrides failed")
		return
	}
	dap := dst.Info()
	sap := dap.Clone()
	sap.SetShape(tmpShape...)
	sap.SetStrides(newStrides)

	diter := flatiter.New(dap)
	siter := flatiter.New(&sap)
	copyIter(dst.Data(), src.Data(), diter, siter)
	return

}
