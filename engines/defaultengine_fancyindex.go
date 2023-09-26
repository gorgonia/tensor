package stdeng

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
)

// SelectByIndices "selects" an index in a given axis, given a list of indices, and puts the selection in the `retVal`.
// `indices` is assumed to be a vector and will be treated as so.
func (e StdEng[DT, T]) SelectByIndices(ctx context.Context, a T, indices tensor.Basic[int], axis int, retVal T) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}
	if axis >= a.Dims() {
		return errors.Errorf(errors.InvalidAxis, axis, a.Dims())
	}
	useIter := a.RequiresIterator() || indices.RequiresIterator() || retVal.RequiresIterator() || !a.DataOrder().HasSameOrder(indices.DataOrder())
	if useIter {
		return errors.Errorf(errors.NYIPR, "SelectByIndices", "StdEng on tensors that require iterators.")
	}

	return e.selectByIdx(axis, indices.Data(), a.Data(), retVal.Data(), a.Info(), retVal.Info())

}

func (e StdEng[DT, T]) selectByIdx(axis int, indices []int, a, retVal []DT, apA, apRet *tensor.AP) (err error) {
	aStrides := apA.Strides()
	rStrides := apRet.Strides()
	aShape := apA.Shape()
	rShape := apRet.Shape()
	axStride := aStrides[axis]
	retStride := rStrides[axis]

	var outerRetStride int
	if axis == 0 {
		// then it's the outermost
		outerRetStride = rStrides[axis] * 2
	} else {
		outerRetStride = rStrides[axis-1]
	}

	srcCoord := make([]int, apA.Dims())
	dstCoord := make([]int, apRet.Dims())

	isInnermost := axis == aShape.Dims()-1
	outer := internal.Prod[int](aShape[:axis])

	var start, dstStart int
	if isInnermost {
		prevAxis := axis - 1
		if prevAxis < 0 {
			// this may be the case if input is a vector
			prevAxis = 0
		}
		prevStride := aStrides[prevAxis]
		retPrevStride := rStrides[prevAxis]
		for i, idx := range indices {
			srcCoord[axis] = idx
			dstCoord[axis] = i
			if start, err = tensor.Ltoi(aShape, aStrides, srcCoord...); err != nil {
				return err
			}
			if dstStart, err = tensor.Ltoi(rShape, rStrides, dstCoord...); err != nil {
				return err
			}
			for o := 0; o < outer; o++ {
				end := start + axStride
				dstEnd := dstStart + retStride

				copy(retVal[dstStart:dstEnd], a[start:end])
				start += prevStride
				dstStart += retPrevStride

			}
		}
		return
	}

	for i, idx := range indices {
		srcCoord[axis] = idx
		dstCoord[axis] = i
		if start, err = tensor.Ltoi(aShape, aStrides, srcCoord...); err != nil {
			return err
		}
		if dstStart, err = tensor.Ltoi(rShape, rStrides, dstCoord...); err != nil {
			return err
		}

		for o := 0; o < outer; o++ {
			end := start + axStride
			dstEnd := dstStart + retStride
			copy(retVal[dstStart:dstEnd], a[start:end])

			start = end + axStride
			dstStart = dstEnd + (outerRetStride - retStride)
		}
	}
	return nil
}

// SelectByIndicesB computes the gradient of the result of `SelectByIndices`.
//
// Currently SelectByIndicesB only supports Dense tensors that do not require the use of iterators.
// Please make a pull request to support tensors that require the use of an iterator to traverse data.
func (e StdNumEngine[DT, T]) SelectByIndicesB(ctx context.Context, input, outGrad T, indices tensor.Basic[int], axis int, retVal T) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}

	// TODO: if input is a scalar...then use slice

	useIter := input.RequiresIterator() || indices.RequiresIterator() || retVal.RequiresIterator() || !input.DataOrder().HasSameOrder(indices.DataOrder())
	if useIter {
		return errors.Errorf(errors.NYIPR, "SelectByIndicesB", "StdEng on tensors that require iterators.")
	}
	e.selectByIndicesB(axis, indices.Data(), outGrad.Data(), retVal.Data(), outGrad.Info(), retVal.Info())
	return nil
}

func (e StdNumEngine[DT, T]) selectByIndicesB(axis int, indices []int, dataB []DT, dataGradA []DT, apB, apRet *tensor.AP) {
	isInnermost := axis == apB.Dims()-1
	outer := internal.Prod[int](apB.Shape()[:axis])
	axStride := apB.Strides()[axis]
	retStride := apRet.Strides()[axis]

	var outerRetStride int
	if axis == 0 {
		outerRetStride = apRet.Strides()[axis] * 2
	} else {
		outerRetStride = apRet.Strides()[axis-1]
	}
	dstCoord := make([]int, apB.Dims())
	srcCoord := make([]int, apRet.Dims())
	if isInnermost {
		prevAxis := axis - 1
		if prevAxis < 0 {
			// that means the input is a vector
			prevAxis = 0
		}
		retPrevStride := apB.Strides()[prevAxis]
		prevStride := apRet.Strides()[prevAxis]

		for i, idx := range indices {
			dstCoord[axis] = idx
			srcCoord[axis] = i
			dstStart, _ := tensor.Ltoi(apB.Shape(), apB.Strides(), dstCoord...)
			start, _ := tensor.Ltoi(apRet.Shape(), apRet.Strides(), srcCoord...)
			for o := 0; o < outer; o++ {
				dstEnd := dstStart + axStride
				end := start + retStride
				execution.AddVV(dataGradA[dstStart:dstEnd], dataB[start:end], dataGradA[dstStart:dstEnd])

				dstStart += prevStride
				start += retPrevStride

			}
		}
		return
	}

	for i, idx := range indices {
		dstCoord[axis] = idx
		srcCoord[axis] = i
		dstStart, _ := tensor.Ltoi(apRet.Shape(), apRet.Strides(), dstCoord...)
		start, _ := tensor.Ltoi(apB.Shape(), apB.Strides(), srcCoord...)

		for o := 0; o < outer; o++ {
			dstEnd := dstStart + axStride
			end := start + retStride

			execution.AddVV(dataGradA[dstStart:dstEnd], dataB[start:end], dataGradA[dstStart:dstEnd])

			dstStart = dstEnd + axStride
			start = end + (outerRetStride - retStride)
		}
	}
}
