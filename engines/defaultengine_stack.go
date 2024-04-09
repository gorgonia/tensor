package stdeng

import (
	"context"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

func (e StdEng[DT, T]) Stack(ctx context.Context, a T, axis int, others ...T) (retVal T, err error) {
	opdims := a.Dims()
	if axis >= opdims+1 {
		err = errors.Errorf(errors.DimMismatch, opdims+1, axis)
		return
	}

	newShape := make(shapes.Shape, opdims+1)
	newShape[axis] = len(others) + 1
	shape := a.Shape()
	var cur int
	for i, s := range shape {
		if i == axis {
			cur++
		}
		newShape[cur] = s
		cur++
	}

	info := a.Info()
	var newStrides []int
	if info.DataOrder().IsColMajor() {
		newStrides = tensor.CalcStridesColMajor(newShape)
	} else {
		newStrides = tensor.CalcStrides(newShape)

	}
	ap := tensor.MakeAP(newShape, newStrides, info.DataOrder(), info.Δ)

	allNoMat := !a.RequiresIterator()
	for _, ot := range others {
		if allNoMat && ot.RequiresIterator() {
			allNoMat = false
		}
	}

	aliker, ok := any(a).(tensor.Aliker[T])
	if !ok {
		var z T
		return retVal, errors.Errorf("Cannot create retVal. Not an Aliker[%T]", z)
	}
	retVal = aliker.Alike(tensor.WithShape(ap.Shape()...), tensor.WithEngine(e))
	//retVal.setAP(&ap) // TODO

	// the "viewStack" method is the more generalized method
	// and will work for all Tensors, regardless of whether it's a view
	// But the simpleStack is faster, and is an optimization

	if allNoMat {
		retVal, err = e.denseSimpleStack(a, retVal, axis, others)
	} else {
		ots := make([]tensor.Basic[DT], 0, len(others))
		for i := range others {
			ots = append(ots, others[i])
		}
		var ret tensor.Basic[DT]
		if ret, err = e.denseViewStack(a, retVal, axis, ots); err != nil {
			return retVal, err
		}
		retVal = ret.(T)
	}
	return
}

func (e StdEng[DT, T]) denseSimpleStack(t, retVal T, axis int, others []T) (T, error) {
	retValData := retVal.Data()
	tData := t.Data()
	switch axis {
	case 0:

		copy(retValData, tData)

		next := len(tData)
		for _, ot := range others {
			otData := ot.Data()
			copy(retValData[next:len(retValData)], otData[:])
			next += len(otData)
		}
	default:
		axisStride := retVal.Info().Strides()[axis]
		batches := len(retValData) / axisStride

		destStart := 0
		start := 0
		end := start + axisStride

		for i := 0; i < batches; i++ {
			copy(retValData[destStart:len(retValData)], tData[start:end])
			for _, ot := range others {
				destStart += axisStride
				otData := ot.Data()
				copy(retValData[destStart:len(retValData)], otData[start:end])
				i++
			}
			destStart += axisStride
			start += axisStride
			end += axisStride
		}
	}
	return retVal, nil
}

func (e StdEng[DT, T]) denseViewStack(t, retVal tensor.Basic[DT], axis int, others []tensor.Basic[DT]) (tensor.Basic[DT], error) {
	retValData := retVal.Data()
	axisStride := retVal.Info().Strides()[axis]
	batches := len(retValData) / axisStride
	it := t.Iterator()
	its := make([]tensor.Iterator, 0, len(others))
	for _, ot := range others {
		its = append(its, ot.Iterator())
	}

	f := func(t tensor.Basic[DT], it Iterator) (last int, err error) {
		for last = 0; last < axisStride; last++ {
			id, err := it.Next()
			if internal.HandleNoOp(err) != nil {
				return -1, errors.Wrap(err, "doViewStack failed")
			}
			if err != nil {
				break
			}
			retValData = append(retValData, t.Data()[id])
		}
		return
	}

	var err error
	for i := 0; i < batches; i++ {
		var last int
		if last, err = f(t, it); err != nil {
			return nil, err
		}
		for j, ot := range others {
			if last, err = f(ot, its[j]); err != nil {
				return nil, err
			}
		}
		_ = last
	}
	return retVal, nil
}
