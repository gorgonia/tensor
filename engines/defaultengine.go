package stdeng

import (
	"context"
	"fmt"
	"reflect"
	"sort"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
)

// StdEng is the standard engine that is supported by gorgonia.org/tensor
type StdEng[DT any, T tensor.Basic[DT]] struct{}

func (e StdEng[DT, T]) AllocAccessible() bool             { return true }
func (e StdEng[DT, T]) Alloc(size int64) (Memory, error)  { return nil, errors.NoOp{} }
func (e StdEng[DT, T]) Free(mem Memory, size int64) error { return nil }
func (e StdEng[DT, T]) Memset(mem Memory, val any) error {
	if ms, ok := mem.(Memsetter); ok {
		return ms.Memset(val)
	}
	return errors.Errorf("Cannot memset %v with StdEng", mem)
}
func (e StdEng[DT, T]) Memclr(mem Memory)                     { panic("NYI") }
func (e StdEng[DT, T]) Memcpy(dst, src Memory) error          { panic("NYI") }
func (e StdEng[DT, T]) Accessible(mem Memory) (Memory, error) { panic("NYI") }
func (e StdEng[DT, T]) WorksWith(flags MemoryFlag, order DataOrder) bool {
	return flags.IsNativelyAccessible()
}

// BasicEng turns an engine that has methods that take a specialized T into one that takes tensor.Basic[DT] as inputs.
func (e StdEng[DT, T]) BasicEng() Engine { return StdEng[DT, tensor.Basic[DT]]{} }

// SliceEq compares if two slices are the same. The datatype must implement an `Eq` method.
func (e StdEng[DT, T]) SliceEq(a, b []DT) bool {
	if internal.SliceEqMeta(a, b) {
		return true
	}

	if len(a) != len(b) {
		return false
	}

	// now to compare it elementwise, we'd have to actually check if DT has an Eq method
	a = a[:len(a)]
	b = b[:len(a)]
	var v DT
	switch any(v).(type) {
	case eqer1[DT]:
		for i := range a {
			v := any(a[i]).(eqer1[DT])
			if !v.Eq(b[i]) {
				return false
			}
		}
	case eqer2:
		for i := range a {
			v := any(a[i]).(eqer2)
			if !v.Eq(b[i]) {
				return false
			}
		}
	case bool:
		a := any(a).([]bool)
		b := any(b).([]bool)
		for i := range a {
			if a[i] != b[i] {
				return false
			}
		}
	default:
		if rv := reflect.ValueOf(v); rv.Comparable() {
			panic("NYI")
		}
		panic(fmt.Sprintf("Cannot call SliceEq on %T - it does not support an `Eq` method and is not a comparable type", v))
	}
	return true
}

func (e StdEng[DT, T]) Transpose(ctx context.Context, t T, expStrides []int) error {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}

	it := t.Iterator()
	data := t.Data()

	out := execution.Transpose(data, it)
	copy(data, out)
	return nil
}

func (e StdEng[DT, T]) Reduce(ctx context.Context, fn any, a T, axis int, defaultValue DT, retVal T) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}

	lastAxis := a.Dims() - 1

	switch {
	case (axis == 0 && a.DataOrder().IsRowMajor()) || ((axis == lastAxis || axis == len(a.Shape())-1) && a.DataOrder().IsColMajor()):
		var size, split int
		if a.DataOrder().IsColMajor() {
			return errors.Errorf("NYI: colmajor")
		}
		size = a.Shape()[0]
		split = a.DataSize() / size

		aData := a.Data()
		retValData := retVal.Data()
		copy(retValData[0:split], aData[0:split])

		switch fn := fn.(type) {
		case func([]DT, []DT):
			execution.ReduceFirstN[DT](a.Data(), retValData, split, size, fn)
		case func(DT, DT) DT:
			execution.ReduceFirst[DT](a.Data(), retValData, split, size, fn)
		case func(DT, DT) (DT, error):
			err = execution.ReduceFirstWithErr[DT](a.Data(), retValData, split, size, fn)
		case tensor.ReductionModule[DT]:
			s := isSameSlice(aData, retValData)
			if fn.ReduceFirstN != nil && !s {
				execution.ReduceFirstN[DT](a.Data(), retValData, split, size, fn.ReduceFirstN)
			} else {
				execution.ReduceFirst[DT](a.Data(), retValData, split, size, fn.Reduce)
			}
		default:
			err = errors.Errorf("Unable to reduce with function of type %T", fn)
		}

	case (axis == lastAxis && a.DataOrder().IsRowMajor()) || (axis == 0 && a.DataOrder().IsColMajor()):
		var dimSize int
		if a.DataOrder().IsColMajor() {
			return errors.Errorf("NYI: colmajor")
		}
		dimSize = a.Shape()[axis]
		switch fn := fn.(type) {
		case func([]DT, DT) DT:
			execution.ReduceLastN[DT](a.Data(), retVal.Data(), dimSize, defaultValue, fn)
		case func(DT, DT) DT:
			execution.ReduceLast[DT](a.Data(), retVal.Data(), dimSize, defaultValue, fn)
		case func(DT, DT) (DT, error):
			err = execution.ReduceLastWithErr[DT](a.Data(), retVal.Data(), dimSize, defaultValue, fn)
		case tensor.ReductionModule[DT]:
			if fn.ReduceLastN != nil {
				execution.ReduceLastN[DT](a.Data(), retVal.Data(), dimSize, defaultValue, fn.ReduceLastN)
			} else {
				execution.ReduceLast[DT](a.Data(), retVal.Data(), dimSize, defaultValue, fn.Reduce)
			}
		default:
			err = errors.Errorf("Unable to reduce last axis with function of type %T", fn)
		}

	default:
		dim0 := a.Shape()[0]
		dimSize := a.Shape()[axis]
		outerStride := a.Strides()[0]
		stride := a.Strides()[axis]
		expected := retVal.Strides()[0]
		switch fn := fn.(type) {
		case func(DT, DT) DT:
			execution.ReduceDefault[DT](a.Data(), retVal.Data(), dim0, dimSize, outerStride, stride, expected, fn)
		case func(DT, DT) (DT, error):
			err = execution.ReduceDefaultWithErr[DT](a.Data(), retVal.Data(), dim0, dimSize, outerStride, stride, expected, fn)
		case tensor.ReductionModule[DT]:
			execution.ReduceDefault[DT](a.Data(), retVal.Data(), dim0, dimSize, outerStride, stride, expected, fn.Reduce)
		default:
			err = errors.Errorf("Unable to reduce axis %d with function of type %T", axis, fn)
		}
	}
	if err == nil {
		// reshape
		newShape := internal.ReduceShape(a.Shape(), axis)
		err = retVal.Reshape(newShape...)
	}
	return
}

func (e StdEng[DT, T]) ReduceAlong(ctx context.Context, mod any, defaultValue DT, a T, retVal T, along ...int) (err error) {

	fn := mod.(tensor.ReductionModule[DT])
	monotonic, incr1 := tensor.IsMonotonicInts(along)
	if (monotonic && incr1 && len(along) == a.Dims()) || len(along) == 0 {
		r := fn.MonotonicReduction(a.Data())
		retVal.Data()[0] = r
	}

	if fn.IsNonCommutative {
		return e.nonCommReduceAlong(ctx, fn, defaultValue, a, retVal, along...)
	}

	var dimsReduced int
	sort.Slice(along, func(i, j int) bool { return along[i] < along[j] })
	input := a
	for _, axis := range along {
		axis -= dimsReduced
		dimsReduced++

		if axis >= retVal.Dims() {
			return errors.Errorf(errors.DimMismatch, retVal.Dims(), axis)
		}

		if err = e.Reduce(ctx, fn, input, axis, defaultValue, retVal); err != nil {
			return err
		}

		input = retVal
	}
	return nil
}

func (e StdEng[DT, T]) nonCommReduceAlong(ctx context.Context, fn tensor.ReductionModule[DT], defaultValue DT, a T, retVal T, along ...int) (err error) {
	track := make([]int, a.Dims())
	input := a
	for _, axis := range along {
		for i := range track {
			if i > axis {
				track[i]++
			}
		}
		axis -= track[axis]

		if axis >= retVal.Dims() {
			return errors.Errorf(errors.DimMismatch, retVal.Dims(), axis)
		}
		if err = e.Reduce(ctx, fn, input, axis, defaultValue, retVal); err != nil {
			return err
		}
		input = retVal
	}
	return nil
}

func (e StdEng[DT, T]) Scan(ctx context.Context, fn any, a, retVal T, axis int) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}
	lastAxis := a.Dims() - 1
	if a.DataOrder().IsColMajor() {
		return errors.Errorf(errors.NYIPR, errors.ThisFn())
	}
	shp := a.Shape()
	strides := a.Strides()
	aData := a.Data()
	retValData := retVal.Data()
	switch {
	case axis == 0:
		// first axis

		dimSize := shp[0]
		stride := strides[0]

		switch fn := fn.(type) {
		case func([]DT, []DT):
			panic("NYI - ScanFirstN is somehow not yet implemented")
		case func(DT, DT) DT:
			execution.ScanFirst(aData, retValData, dimSize, stride, fn)
		default:
			err = errors.Errorf("Unable to scan on axis %d with function of type %T", axis, fn)
		}
	case axis == lastAxis:
		dimSize := shp[axis]
		switch fn := fn.(type) {
		case func([]DT, []DT):
			execution.ScanLastN(aData, retValData, dimSize, fn)
		case func(DT, DT) DT:
			execution.ScanLast(aData, retValData, dimSize, fn)
		default:
			err = errors.Errorf("Unable to scan on axis %d with function type %T", axis, fn)
		}
	default:
		dim0 := shp[0]
		dimSize := shp[axis]
		outerStride := strides[0]
		stride := strides[axis]
		expected := retVal.Strides()[0]
		switch fn := fn.(type) {
		case func(DT, DT) DT:
			execution.ScanDefault(aData, retVal, dim0, dimSize, outerStride, stride, expected, fn)
		default:
			err = errors.Errorf("Unable to scan on axis %d with function type %T", axis, fn)
		}
	}
	return
}

func (e StdEng[DT, T]) Map(ctx context.Context, fn any, a T, retVal T) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}
	if a.RequiresIterator() || retVal.RequiresIterator() {
		ait := a.Iterator()
		rit := retVal.Iterator()
		switch fn := fn.(type) {
		case func(DT) DT:
			return execution.MapIter(fn, a.Data(), retVal.Data(), ait, rit)
		case func(DT) (DT, error):
			return execution.MapIterWithErr(fn, a.Data(), retVal.Data(), ait, rit)
		default:
			return errors.Errorf("Unable to map fn %T ", fn)
		}

	}
	switch fn := fn.(type) {
	case func(DT) DT:
		return execution.Map(fn, a.Data(), retVal.Data())
	case func(DT) (DT, error):
		return execution.MapWithErr(fn, a.Data(), retVal.Data())
	default:
		return errors.Errorf("Unable to map fn %T", fn)

	}
}

// DotIter is APL's `.`
func (e StdEng[DT, T]) DotIter(ctx context.Context, reduceWithFn, elwiseFn func(DT, DT) DT, a, b, retVal T) (err error) {
	/*
	   pseudo algorithm:

	   1. reshape `a` and `b` to matrices
	   2. call execution.DotIterA
	*/
	// TODO: this assumes that this is dense
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}

	// compute shape
	aShape := asMat(a.Shape(), a.Dims()-1, true)
	bShape := asMat(b.Shape(), 1, true)
	cShape := shapes.Shape{aShape[0], bShape[1]}

	m, n, k, lda, ldb, ldc, tA, tB := e.MatMulHelper(a, b, retVal, aShape, bShape, cShape)

	return execution.GeDOR(reduceWithFn, elwiseFn, tA, tB, m, n, k, a.Data(), lda, b.Data(), ldb, retVal.Data(), ldc)
}

func (e StdEng[DT, T]) Concat(ctx context.Context, a T, axis int, others ...T) (retVal T, err error) {
	panic("NYI")
}

func (e StdEng[DT, T]) Stack(ctx context.Context, a T, axis int, others ...T) (retVal T, err error) {
	panic("NYI")
}
