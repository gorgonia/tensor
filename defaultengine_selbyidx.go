package tensor

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"

	"reflect"
)

func (e StdEng) SelectByIndices(a, b Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !b.Shape().IsVectorLike() {
		return nil, errors.Errorf("Expected indices to be a vector. Got %v instead", b.Shape())
	}
	if b.Dtype() != Int {
		return nil, errors.Errorf("Expected indices to be a vector of ints. Got %v instead", b.Dtype())
	}

	// if b is a scalar, then use Slice
	if a.Shape().IsScalarEquiv() {
		slices := make([]Slice, a.Shape().Dims())
		slices[axis] = ss(b.Data().([]int)[0])
		return a.Slice(slices...)
	}

	expectedShape := a.Shape().Clone()
	expectedShape[axis] = b.Shape().TotalSize()

	var reuse DenseTensor
	var safe, toReuse, _ bool
	if reuse, safe, toReuse, _, _, err = handleFuncOpts(expectedShape, a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if safe || !toReuse && reuse == nil && safe {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(a.Dtype()))
	}

	if !safe {
		if a.Shape()[axis] != b.Shape().TotalSize() {
			expected := a.Shape().Clone()
			expected[axis] = b.Shape().TotalSize()
			return nil, errors.Errorf("Expected a safe resuse to have the same shape as the expected shape of the result: %v. The input a has %v ", expected, a.Shape())
		}

		reuse = a.(DenseTensor)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, _, err = prepDataVV(a, b, reuse); err != nil {
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
		prevStride := apA.strides[axis-1]
		retPrevStride := apRet.strides[axis-1]
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

// SelectByIndicesB is the backwards function of SelectByIndices.
func (e StdEng) SelectByIndicesB(a, b, indices Tensor, axis int, opts ...FuncOpt) (retVal Tensor, err error) {
	if !indices.Shape().IsVectorLike() {
		return nil, errors.Errorf("Expected indices to be a vector. Got %v instead", b.Shape())
	}
	if indices.Dtype() != Int {
		return nil, errors.Errorf("Expected indices to be a vector of ints. Got %v instead", b.Dtype())
	}

	// if b is a scalar, then use Slice
	if a.Shape().IsScalarEquiv() {
		slices := make([]Slice, a.Shape().Dims())
		slices[axis] = ss(b.Data().([]int)[0])
		return a.Slice(slices...)
	}

	expectedShape := a.Shape().Clone()

	var reuse DenseTensor
	var _, toReuse, _ bool
	if reuse, _, toReuse, _, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if !toReuse && reuse == nil {
		// create reuse
		reuse = New(WithShape(expectedShape...), Of(a.Dtype()))
	}

	typ := a.Dtype().Type
	var _, dataB, dataReuse *storage.Header
	var _, bit, iit Iterator
	var useIter bool
	if _, dataB, dataReuse, _, bit, iit, useIter, _, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.SelectByIndicesB")
	}

	if useIter {
		e.iterSelectByIndicesB(axis, dataB, dataReuse, bit, iit)
		//TODO
		return
	}

	e.selectByIndicesB(axis, indices.Data().([]int), typ, dataB, dataReuse, b.(*Dense).AP, reuse.(*Dense).AP)

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
		retPrevStride := apB.strides[axis-1]
		prevStride := apRet.strides[axis-1]
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
		dstStart, _ := Ltoi(apB.shape, apB.strides, dstCoord...)
		start, _ := Ltoi(apRet.shape, apRet.strides, srcCoord...)

		for o := 0; o < outer; o++ {
			dstEnd := dstStart + axStride
			end := start + retStride

			e.E.AddSliced(typ, dataGradA, dstStart, dstEnd, dataB, start, end)

			dstStart = dstEnd + axStride
			start = end + (outerRetStride - retStride)
		}
	}
}
