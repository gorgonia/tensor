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
func (e StdEng[DT, T]) Memclr(mem Memory) {
	var z DT
	acc, ok := mem.(tensor.RawAccessor[DT])
	if !ok {
		panic("Cannot Memclr non RawAccessor memories")
	}
	data := acc.Data()
	for i := range data {
		data[i] = z
	}
}
func (e StdEng[DT, T]) Memcpy(dst, src Memory) error {
	switch dst := dst.(type) {
	case tensor.RawAccessor[DT]:
		switch src := src.(type) {
		case tensor.RawAccessor[DT]:
			copy(dst.Data(), src.Data())
			return nil

		}
	}
	return errors.Errorf("Cannot Memcpy Memory of %T and %T", dst, src)
}

func (e StdEng[DT, T]) Accessible(mem Memory) (Memory, error) { return mem, nil }
func (e StdEng[DT, T]) WorksWith(flags MemoryFlag, order DataOrder) bool {
	return flags.IsNativelyAccessible()
}

// Workhorse returns the engine that will actually do all the work (in this case, itself).
func (e StdEng[DT, T]) Workhorse() Engine { return e }

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

func (e StdEng[DT, T]) Transpose(ctx context.Context, t tensor.Basic[DT], expStrides []int) error {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}

	it := t.Iterator()
	data := t.Data()

	out := execution.Transpose(data, it)
	copy(data, out)
	return nil
}

func (e StdEng[DT, T]) Copy(ctx context.Context, dst, src T) error {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}
	dit, sit, useIter, err := PrepDataUnary[DT](dst, src) // src is not actually a reuse, but eh.
	if err != nil {
		return err
	}
	if useIter {
		return execution.CopyIter(dst.Data(), src.Data(), dit, sit)
	}
	copy(dst.Data(), src.Data())
	return nil
}

func (e StdEng[DT, T]) Reduce(ctx context.Context, fn any, a tensor.Basic[DT], axis int, defaultValue DT, retVal tensor.Basic[DT]) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}

	lastAxis := a.Dims() - 1
	if a.DataOrder().IsColMajor() {
		// case axis == lastAxis && a.DataOrder().IsColMajor():
		// 	should be the same as case axis ==0 && a.DataOrder().IsRowMajor().
		// case axis ==0 && a.DataOrder().IsColMajor:
		// 	should be the same case as axis == lastAxis && a.DataOrder.IsRowMajor()
		return errors.Errorf(errors.NYIPR, "ColMajor", errors.ThisFn())
	}

	shp := a.Shape()
	strides := a.Strides()

	aData := a.Data()
	retValData := retVal.Data()
	switch axis {
	case 0:
		var size, split int
		size = shp[0]
		split = a.DataSize() / size
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

	case lastAxis:
		dimSize := shp[axis]
		switch fn := fn.(type) {
		case func([]DT, DT) DT:
			execution.ReduceLastN[DT](aData, retValData, dimSize, defaultValue, fn)
		case func(DT, DT) DT:
			execution.ReduceLast[DT](aData, retValData, dimSize, defaultValue, fn)
		case func(DT, DT) (DT, error):
			err = execution.ReduceLastWithErr[DT](aData, retValData, dimSize, defaultValue, fn)
		case tensor.ReductionModule[DT]:
			if fn.ReduceLastN != nil {
				execution.ReduceLastN[DT](aData, retValData, dimSize, defaultValue, fn.ReduceLastN)
			} else {
				execution.ReduceLast[DT](aData, retValData, dimSize, defaultValue, fn.Reduce)
			}
		default:
			err = errors.Errorf("Unable to reduce last axis with function of type %T", fn)
		}

	default:
		dim0 := shp[0]
		dimSize := shp[axis]
		outerStride := strides[0]
		stride := strides[axis]
		expected := retVal.Strides()[0]
		switch fn := fn.(type) {
		case func(DT, DT) DT:
			execution.ReduceDefault[DT](aData, retValData, dim0, dimSize, outerStride, stride, expected, fn)
		case func(DT, DT) (DT, error):
			err = execution.ReduceDefaultWithErr[DT](aData, retValData, dim0, dimSize, outerStride, stride, expected, fn)
		case tensor.ReductionModule[DT]:
			execution.ReduceDefault[DT](aData, retValData, dim0, dimSize, outerStride, stride, expected, fn.Reduce)
		default:
			err = errors.Errorf("Unable to reduce axis %d with function of type %T", axis, fn)
		}
	}
	if err == nil {
		// reshape
		newShape := internal.ReduceShape(shp, axis)
		err = retVal.Reshape(newShape...)
	}
	return
}

func (e StdEng[DT, T]) ReduceAlong(ctx context.Context, mod any, defaultValue DT, a tensor.Basic[DT], retVal tensor.Basic[DT], along ...int) (err error) {
	fn := mod.(tensor.ReductionModule[DT])
	if fn.IsNonCommutative {
		return e.nonCommReduceAlong(ctx, fn, defaultValue, a, retVal, along...)
	}

	monotonic, incr1 := tensor.IsMonotonicInts(along)
	if (monotonic && incr1 && len(along) == a.Dims()) || len(along) == 0 {
		if fn.MonotonicReduction == nil {
			fn.MonotonicReduction = internal.MakeMonotonicReduction(fn.Reduce, defaultValue)
		}
		r := fn.MonotonicReduction(a.Data())
		retVal.Data()[0] = r

		return retVal.Reshape() // it's gonna be scalar,bro
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

func (e StdEng[DT, T]) nonCommReduceAlong(ctx context.Context, fn tensor.ReductionModule[DT], defaultValue DT, a tensor.Basic[DT], retVal tensor.Basic[DT], along ...int) (err error) {
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

func (e StdEng[DT, T]) Scan(ctx context.Context, fn any, a tensor.Basic[DT], axis int, retVal tensor.Basic[DT]) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}
	lastAxis := a.Dims() - 1
	if a.DataOrder().IsColMajor() {
		return errors.Errorf(errors.NYIPR, "ColMajor", errors.ThisFn())
	}
	shp := a.Shape()
	strides := a.Strides()
	aData := a.Data()
	retValData := retVal.Data()
	switch axis {
	case 0:
		// first axis
		dimSize := shp[0]
		stride := strides[0]

		switch fn := fn.(type) {
		case func([]DT, []DT):
			panic("NYI - ScanFirstN is somehow not yet implemented")
		case func(DT, DT) DT:
			// log.Printf("dimSize %v stride %v", dimSize, stride)
			execution.ScanFirst(aData, retValData, dimSize, stride, fn)
		default:
			err = errors.Errorf("Unable to scan on axis %d with function of type %T", axis, fn)
		}
	case lastAxis:
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
		expected := 1 //retVal.Strides()[0]
		switch fn := fn.(type) {
		case func(DT, DT) DT:
			// log.Printf("didm0 %v dimSize %v, outerStrides %v stride %v, expected %v", dim0, dimSize, outerStride, stride, expected)
			execution.ScanDefault(aData, retValData, dim0, dimSize, outerStride, stride, expected, fn)
		default:
			err = errors.Errorf("Unable to scan on axis %d with function type %T", axis, fn)
		}
	}
	return
}

func (e StdEng[DT, T]) Map(ctx context.Context, fn any, a tensor.Basic[DT], retVal tensor.Basic[DT]) (err error) {
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
func (e StdEng[DT, T]) DotIter(ctx context.Context, reduceWithFn, elwiseFn func(DT, DT) DT, a, b, retVal tensor.Basic[DT]) (err error) {
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

func (e StdEng[DT, T]) Concat(ctx context.Context, a T, axis int, others ...tensor.Basic[DT]) (retVal T, err error) {
	ss := make([]shapes.Shapelike, len(others))
	for i, o := range others {
		ss[i] = o.Shape()
	}

	newShapelike, err := a.Shape().Concat(shapes.Axis(axis), ss...)
	if err != nil {
		return retVal, err
	}
	newShape := newShapelike.(shapes.Shape)
	aliker, ok := any(a).(tensor.Aliker[T])
	if !ok {
		return retVal, errors.Errorf("Unable to concat %T", a)
	}
	retVal = aliker.Alike(tensor.WithShape(newShape...), tensor.WithEngine(e))
	slicer := any(retVal).(tensor.Slicer[T])
	// if masked{
	// retVal.makeMask() // this should be handled by the tensor.Aliker
	//}

	all := make([]tensor.Basic[DT], len(others)+1)
	all[0] = a
	copy(all[1:], others)

	// TODO: OPIMIZATION
	// When (axis == 0 && a is row major and all others is row major) || (axis == last axis of A && all tensors are colmajor)
	// just flat copy
	//

	isOuter := axis == 0
	isInner := axis == (a.Dims() - 1)

	// special case
	var start, end int
	for _, x := range all {
		xshp := x.Shape()
		end += xshp[axis]
		slices := make([]SliceRange, axis+1)
		slices[axis] = shapes.S(start, end)

		var v T
		if v, err = slicer.Slice(slices...); err != nil {
			return
		}
		shp := v.Shape()
		// keep dims after slicing
		switch {
		case shp.IsVector() && xshp.IsMatrix() && axis == 0:
			v.Reshape(shp[0], 1)
		case xshp.IsRowVec() && axis == 0:
			x.Reshape(xshp[1])
		case shp.IsScalarEquiv() && xshp.IsScalarEquiv():
			copy(v.Data(), x.Data())
			// if mt, ok := T.(MaskedTensor); ok {
			// 	copy(v.mask, mt.Mask())
			// }
			start = end
			continue
		default:
			diff := retVal.Shape().Dims() - v.Shape().Dims()
			if diff > 0 && isOuter {
				newShape := make(shapes.Shape, v.Shape().Dims()+diff)
				for i := 0; i < diff; i++ {
					newShape[i] = 1
				}
				copy(newShape[diff:], v.Shape())
				v.Reshape(newShape...)
			} else if diff > 0 && isInner {
				newShape := v.Shape().Clone()
				newStrides := v.Strides()
				for i := 0; i < diff; i++ {
					newShape = append(newShape, 1)
					newStrides = append(newStrides, 1)
				}
				v.Info().SetShape(newShape...)
				v.Info().SetStrides(newStrides)
			} else if xshp[axis] == 1 {
				if err := v.Unsqueeze(axis); err != nil {
					return retVal, errors.Wrapf(err, "Unable to keep dims after slicing a shape %v on axis %d where the size is 1", x.Shape(), axis)
				}
			}
		}

		// var vmask, Tmask []bool
		// vmask = v.mask
		// v.mask = nil
		// if mt, ok := T.(MaskedTensor); ok && mt.IsMasked() {
		// 	Tmask = mt.Mask()
		// 	mt.SetMask(nil)

		// }

		if err = assign[DT](v, x); err != nil {
			return retVal, errors.Wrapf(err, errors.OpFail, "Concat - assign stage")
		}

		// // if it's a masked tensor, we copy the mask as well
		// if Tmask != nil {
		// 	if vmask != nil {
		// 		if cap(vmask) < len(Tmask) {
		// 			vmask2 := make([]bool, len(Tmask))
		// 			copy(vmask2, vmask)
		// 			vmask = vmask2
		// 		}
		// 		copy(vmask, Tmask)
		// 		v.SetMask(vmask)
		// 	}
		// 	// mt.SetMask(Tmask)
		// }

		start = end

	}
	return retVal, nil
}
