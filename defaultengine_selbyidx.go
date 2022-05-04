package tensor

import (
	"context"

	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"

	"reflect"
)

// SelectByIndices selects the values given the in `indices` tensor.
//
// Currently SelectByIndices only supports Dense tensors that do not require the use of iterators.
// Please make a pull request to support tensors that require the use of an iterator to traverse data.
func (e StdEng) SelectByIndices(a, indices Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !indices.Shape().IsVectorLike() {
		return nil, errors.Errorf("Expected indices to be a vector. Got %v instead", indices.Shape())
	}
	if indices.Dtype() != Int {
		return nil, errors.Errorf("Expected indices to be a vector of ints. Got %v instead", indices.Dtype())
	}

	// if b is a scalar, then use Slice
	if a.Shape().IsScalarEquiv() {
		slices := make([]Slice, a.Shape().Dims())
		slices[axis] = ss(getInts(indices)[0])
		return a.Slice(slices...)
	}

	expectedShape := a.Shape().Clone()
	expectedShape[axis] = indices.Shape().TotalSize()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	var ctx context.Context
	if ctx, reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = handleCtx(ctx); err != nil {
		return nil, err // will be noopError{}, no need to wrap.
	}
	if safe || !toReuse && reuse == nil && safe {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(a.Dtype()))
	}

	if !safe {
		if a.Shape()[axis] != indices.Shape().TotalSize() {
			expected := a.Shape().Clone()
			expected[axis] = indices.Shape().TotalSize()
			return nil, errors.Errorf("Expected a safe resuse to have the same shape as the expected shape of the result: %v. The input a has %v ", expected, a.Shape())
		}

		reuse = a.(DenseTensor)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, _, err = prepDataVV(a, indices, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Add")
	}

	if useIter {
		e.iterSelectByIdx(axis, dataA, dataB, dataReuse, ait, bit, iit)
		//TODO
		return
	}

	e.selectByIdx(axis, dataB.Ints(), typ, dataA, dataReuse, a.(*Dense).AP, reuse.(*Dense).AP)
	return reuse, nil
}

func (e StdEng) iterSelectByIdx(axis int, dataA, dataB, dataReuse *storage.Header, ait, bit, iit Iterator) {
	panic("iterSelectByIdx is not yet implemented")
}

func (e StdEng) selectByIdx(axis int, indices []int, typ reflect.Type, dataA, dataRetVal *storage.Header, apA, apRet AP) {
	isInnermost := axis == apA.shape.Dims()-1

	outer := ProdInts(apA.shape[:axis])

	axStride := apA.strides[axis]
	retStride := apRet.strides[axis]
	var outerRetStride int
	if axis == 0 {
		// then it's the outermost
		outerRetStride = apRet.strides[axis] * 2
	} else {
		outerRetStride = apRet.strides[axis-1]
	}

	srcCoord := make([]int, apA.shape.Dims())
	dstCoord := make([]int, apRet.shape.Dims())

	if isInnermost {
		prevAxis := axis - 1
		if prevAxis < 0 {
			// this may be the case if input is a vector
			prevAxis = 0
		}
		prevStride := apA.strides[prevAxis]
		retPrevStride := apRet.strides[prevAxis]
		for i, idx := range indices {
			srcCoord[axis] = idx
			dstCoord[axis] = i
			start, _ := Ltoi(apA.shape, apA.strides, srcCoord...)
			dstStart, _ := Ltoi(apRet.shape, apRet.strides, dstCoord...)
			for o := 0; o < outer; o++ {
				end := start + axStride
				dstEnd := dstStart + retStride

				storage.CopySliced(typ, dataRetVal, dstStart, dstEnd, dataA, start, end)

				start += prevStride
				dstStart += retPrevStride

			}
		}
		return
	}

	for i, idx := range indices {
		srcCoord[axis] = idx
		dstCoord[axis] = i
		start, _ := Ltoi(apA.shape, apA.strides, srcCoord...)
		dstStart, _ := Ltoi(apRet.shape, apRet.strides, dstCoord...)

		for o := 0; o < outer; o++ {
			end := start + axStride
			dstEnd := dstStart + retStride

			storage.CopySliced(typ, dataRetVal, dstStart, dstEnd, dataA, start, end)

			start = end + axStride
			dstStart = dstEnd + (outerRetStride - retStride)
		}
	}
}

// SelectByIndicesB computes the gradient of the result of `SelectByIndices`.
//
// Currently SelectByIndicesB only supports Dense tensors that do not require the use of iterators.
// Please make a pull request to support tensors that require the use of an iterator to traverse data.
func (e StdEng) SelectByIndicesB(input, outGrad, indices Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !indices.Shape().IsVectorLike() {
		return nil, errors.Errorf("Expected indices to be a vector. Got %v instead", outGrad.Shape())
	}
	if indices.Dtype() != Int {
		return nil, errors.Errorf("Expected indices to be a vector of ints. Got %v instead", outGrad.Dtype())
	}

	// if b is a scalar, then use Slice
	if input.Shape().IsScalarEquiv() {
		slices := make([]Slice, input.Shape().Dims())
		slices[axis] = ss(outGrad.Data().([]int)[0])
		return input.Slice(slices...)
	}

	expectedShape := input.Shape().Clone()

	var reuse DenseTensor
	var _, toReuse, _ bool
	var ctx context.Context
	if ctx, reuse, _, toReuse, _, _, err = handleFuncOpts(input.Shape(), input.Dtype(), input.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = handleCtx(ctx); err != nil {
		return nil, err // will be noopError{}, no need to wrap.
	}
	if !toReuse && reuse == nil {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(input.Dtype()))
	}

	typ := input.Dtype().Type
	var _, dataB, dataReuse *storage.Header
	var _, bit, iit Iterator
	var useIter bool
	if _, dataB, dataReuse, _, bit, iit, useIter, _, err = prepDataVV(input, outGrad, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.SelectByIndicesB")
	}

	if useIter {
		e.iterSelectByIndicesB(axis, dataB, dataReuse, bit, iit)
		//TODO
		return
	}

	e.selectByIndicesB(axis, getInts(indices), typ, dataB, dataReuse, outGrad.(*Dense).AP, reuse.(*Dense).AP)

	return reuse, nil
}

func (e StdEng) iterSelectByIndicesB(axis int, dataB, dataGradA *storage.Header, bit, iit Iterator) {
	panic("iterSelectByIndicesB not implemented yet")
}

func (e StdEng) selectByIndicesB(axis int, indices []int, typ reflect.Type, dataB, dataGradA *storage.Header, apB, apRet AP) {
	isInnermost := axis == apB.shape.Dims()-1

	outer := ProdInts(apB.shape[:axis])

	axStride := apB.strides[axis]
	retStride := apRet.strides[axis]
	var outerRetStride int
	if axis == 0 {
		outerRetStride = apRet.strides[axis] * 2
	} else {
		outerRetStride = apRet.strides[axis-1]
	}

	dstCoord := make([]int, apB.shape.Dims())
	srcCoord := make([]int, apRet.shape.Dims())

	if isInnermost {
		prevAxis := axis - 1
		if prevAxis < 0 {
			// this may be the case if input is a vector
			prevAxis = 0
		}
		retPrevStride := apB.strides[prevAxis]
		prevStride := apRet.strides[prevAxis]
		for i, idx := range indices {
			dstCoord[axis] = idx
			srcCoord[axis] = i
			dstStart, _ := Ltoi(apB.shape, apB.strides, dstCoord...)
			start, _ := Ltoi(apRet.shape, apRet.strides, srcCoord...)
			for o := 0; o < outer; o++ {
				dstEnd := dstStart + axStride
				end := start + retStride

				e.E.AddSliced(typ, dataGradA, dstStart, dstEnd, dataB, start, end)

				dstStart += prevStride
				start += retPrevStride

			}
		}
		return
	}

	for i, idx := range indices {
		dstCoord[axis] = idx
		srcCoord[axis] = i
		dstStart, _ := Ltoi(apRet.shape, apRet.strides, dstCoord...)
		start, _ := Ltoi(apB.shape, apB.strides, srcCoord...)

		for o := 0; o < outer; o++ {
			dstEnd := dstStart + axStride
			end := start + retStride

			e.E.AddSliced(typ, dataGradA, dstStart, dstEnd, dataB, start, end)

			dstStart = dstEnd + axStride
			start = end + (outerRetStride - retStride)
		}
	}
}
