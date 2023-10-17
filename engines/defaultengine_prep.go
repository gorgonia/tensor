package stdeng

import (
	"context"

	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	gutils "gorgonia.org/tensor/internal/utils"
)

// some constant error messages
const (
	cannotUseReuse   = "Cannot use `reuse`."
	cannotUseUnsafe  = "Cannot use unsafe as an option."
	cannotPrepRepeat = "Cannot prepare tensors for `Repeat`"
)

func checkArrayShape(a tensor.DataSizer, expShape shapes.Shape) error {
	dataSize := a.DataSize()
	expSize := expShape.TotalSize()
	if dataSize < expSize && !expShape.IsScalar() {
		return errors.Errorf(errors.ArrayMismatch, dataSize, expSize)
	}
	return nil
}

func checkDtype(expDtype dtype.Dtype, got dtype.Dtype) error {
	if expDtype == nil {
		return nil
	}
	if !got.Eq(expDtype) {
		return errors.Errorf(errors.DtypeError, expDtype, got)
	}
	return nil
}

func (e StdEng[DT, T]) HandleFuncOptsDesc(a tensor.Basic[DT], expShape shapes.Shape, opts ...FuncOpt) (retVal DescWithStorage, fo Option, err error) {
	fo = ParseFuncOpts(opts...)

	reuseAny := fo.Reuse
	toReuse := reuseAny != nil
	safe := fo.Safe()

	asType := fo.AsType

	var ok bool
	switch {
	case toReuse:
		if retVal, ok = reuseAny.(DescWithStorage); !ok {
			err = errors.Wrap(errors.Errorf(errors.TypeError, retVal, reuseAny), cannotUseReuse)
			return
		}
		// if asType is not nil, then we would expect that the reuse should be of the same dtype
		if err = checkDtype(asType, retVal.Dtype()); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		// check whether we can access data in `retVal`, because the default engines only handles data that Go can access.
		if !retVal.IsNativelyAccessible() {
			err = errors.Wrap(errors.Errorf(errors.InaccessibleData, retVal), cannotUseReuse)
			return
		}

		// restore the state to the original
		retVal.Restore()

		// if the reuse tensor has a smaller array size than the expected array size, then we cannot proceed
		if err = checkArrayShape(retVal, expShape); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		// naughty...
		if !retVal.Shape().Eq(expShape) {
			if err = retVal.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}

		// if the reuse is the target of an increment operation, then we don't set the data order.
		if !fo.Incr {
			retVal.SetDataOrder(a.DataOrder())
		}
	case !safe:
		// if asType is not nil, then we would expect that the reuse should be of the same dtype
		if err = checkDtype(asType, retVal.Dtype()); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		// if the reuse tensor has a smaller array size than the expected array size, we cannot use unsafe.
		if err = checkArrayShape(a, expShape); err != nil {
			err = errors.Wrap(err, cannotUseUnsafe)
			return
		}

		// we reshape the value to the expected shape
		if !a.Shape().Eq(expShape) {
			if err = a.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}

		retVal = a
	default:
		// safe
		if asType != nil {
			retVal = a.AlikeAsType(asType, WithShape(expShape...))
			return
		}

		retVal = a.AlikeAsDescWithStorage(WithShape(expShape...))
	}
	return
}

// HandleFuncOpts is the usual way you would handle FuncOpts. It returns the "retVal" tensor, which is either a reuse tensor passed in from a FuncOpt
// or one that is created by using `Alike`.
func (e StdEng[DT, T]) HandleFuncOpts(a tensor.Basic[DT], expShape shapes.Shape, opts ...FuncOpt) (retVal tensor.Basic[DT], fo Option, err error) {
	fo = ParseFuncOpts(opts...)

	reuseAny := fo.Reuse
	toReuse := reuseAny != nil
	safe := fo.Safe()

	var ok bool
	var empty tensor.Basic[DT]

	if fo.AsType != nil {
		err = errors.Errorf("Cannot use HandleFuncOptsSpecialized with a return tensor of a different dtype %v.")
		return
	}

	switch {
	case toReuse:
		if retVal, ok = reuseAny.(tensor.Basic[DT]); !ok {
			err = errors.Wrap(errors.Errorf(errors.TypeError, empty, reuseAny), cannotUseReuse)
			return
		}

		if !retVal.IsNativelyAccessible() {
			err = errors.Wrap(errors.Errorf(errors.InaccessibleData, retVal), cannotUseReuse)
			return
		}

		// restore the state to the original
		retVal.Restore()

		if err = checkArrayShape(retVal, expShape); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		if !retVal.Shape().Eq(expShape) {
			if err = retVal.Reshape(expShape...); err != nil {
				return
			}
		}

		if !fo.Incr {
			retVal.SetDataOrder(a.DataOrder())
		}
	case !safe:
		// if the reuse tensor has a smaller array size than the expected array size, we cannot use unsafe.
		if err = checkArrayShape(a, expShape); err != nil {
			err = errors.Wrap(err, cannotUseUnsafe)
			return
		}

		// we reshape the value to the expected shape
		if !a.Shape().Eq(expShape) {
			if err = a.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}
		retVal = a
	default:
		// safe
		retVal = a.AlikeAsBasic(WithShape(expShape...))
	}
	return
}

// HandleFuncOptsSpecialized handles specialized function options. As such it doesn't handle creating retVal of different dtypes.
func (e StdEng[DT, T]) HandleFuncOptsSpecialized(a T, expShape shapes.Shape, opts ...FuncOpt) (retVal T, fo Option, err error) {
	fo = ParseFuncOpts(opts...)

	reuseAny := fo.Reuse
	toReuse := reuseAny != nil
	safe := fo.Safe()

	var ok bool
	var empty T

	if fo.AsType != nil {
		err = errors.Errorf("Cannot use HandleFuncOptsSpecialized with a return tensor of a different dtype %v.")
		return
	}

	switch {
	case toReuse:
		if retVal, ok = reuseAny.(T); !ok {
			err = errors.Wrap(errors.Errorf(errors.TypeError, empty, reuseAny), cannotUseReuse)
			return
		}

		if !retVal.IsNativelyAccessible() {
			err = errors.Wrap(errors.Errorf(errors.InaccessibleData, retVal), cannotUseReuse)
			return
		}

		// restore the state to the original
		retVal.Restore()

		if err = checkArrayShape(retVal, expShape); err != nil {
			err = errors.Wrap(err, cannotUseReuse)
			return
		}

		if !retVal.Shape().Eq(expShape) {
			if err = retVal.Reshape(expShape...); err != nil {
				return
			}
		}

		if !fo.Incr {
			retVal.SetDataOrder(a.DataOrder())
		}
	case !safe:
		// if the reuse tensor has a smaller array size than the expected array size, we cannot use unsafe.
		if err = checkArrayShape(a, expShape); err != nil {
			err = errors.Wrap(err, cannotUseUnsafe)
			return
		}

		// we reshape the value to the expected shape
		if !a.Shape().Eq(expShape) {
			if err = a.Reshape(expShape...); err != nil {
				err = errors.Wrap(err, cannotUseReuse)
				return
			}
		}
		retVal = a
	default:
		// safe
		aliker, ok := any(a).(tensor.Aliker[T])
		if !ok {
			return retVal, fo, errors.Errorf("Tensor does not support aliker")
		}
		retVal = aliker.Alike(WithShape(expShape...))
	}
	return
}

func (e StdEng[DT, T]) PrepRepeat(a T, axis int, repeats []int, opts ...FuncOpt) (ctx context.Context, retVal T, newAxis, size int, newRepeats []int, err error) {
	var newShapelike shapes.Shapelike
	newShapelike, newRepeats, size, err = a.Shape().Repeat(shapes.Axis(axis), repeats...)
	if err != nil {
		return
	}
	newShape := newShapelike.(shapes.Shape)
	newAxis = axis
	if axis == AllAxes {
		newAxis = 0
	}

	// now  we have the expected shape, handle the FuncOpts
	var fo Option
	if retVal, fo, err = e.HandleFuncOptsSpecialized(a, newShape, opts...); err != nil {
		err = errors.Wrapf(err, cannotPrepRepeat)
		return
	}
	return fo.Ctx, retVal, newAxis, size, newRepeats, nil
}

// PrepReduce is a helper function to help reductions
func (e StdEng[DT, T]) PrepReduce(a T, opts ...FuncOpt) (ctx context.Context, axes []int, retVal T, err error) {
	// compute the largest shape after reduction
	// for example:
	//
	// Given a tensor of shape (2, 3, 4) and the reduction axes are []int{2,0}
	// the final shape will be (3).
	//
	// There are TWO reduction processes possible:
	// 	commutative reduction
	//	non-commutative reduction
	//
	// In the commutative case, the axes will be sorted - so the new axes are []int{0, 2}
	// In the non-commutative case, the axes will not be sorted.
	//
	// Now let's consider what happens in the commutative case:
	//	1. (2, 3, 4) -> (3, 4)
	// 	2. (3, 4) -> (3,)
	//
	// Because `PrepReduction` has no idea what the reduction function is going to be, we cannot just take
	// axes[0] and find the reduced shape.
	//
	// In this case, axes[0] would be 2 (this is pre-sorting, remember?)
	// so the new shape will be (2, 3). A new reuse tensor of shape (2,3) will be created.
	// When the engine reduces, it'll reduce on a sorted axes, trying to reduce on axis 0 first.
	// The expected shape of reduction on the first axis is: (3, 4), so the expected size of the reuse tensor
	// is at least 3×4 = 12. However, we created a reuse tensor of shape (2,3)!
	//
	// So, the solution is to find the axis that has the smallest dim size, and then we reduce that one.
	// In this case, we search the axes([]int{2,0}) of the shape (2, 3, 4), which yields axis 0 having the smallest
	// dimension, and so we compute the new shape to be at least that.

	shp := a.Shape()
	var min int = 99999999999
	var minAx int
	for _, axis := range axes {
		if shp[axis] < min {
			min = shp[axis]
			minAx = axis
		}
	}
	newShape := internal.ReduceShape(shp, minAx)

	// now  we have the expected shape, handle the FuncOpts
	var fo Option
	if retVal, fo, err = e.HandleFuncOptsSpecialized(a, newShape, opts...); err != nil {
		return ctx, axes, retVal, errors.Wrapf(err, "Unable to PrepReduce")
	}

	ctx = fo.Ctx
	axes = fo.Along
	toReuse := fo.Reuse != nil
	safe := fo.Safe()

	if len(axes) == 0 {
		axes = gutils.Range[int](0, a.Dims())
	}

	for _, axis := range axes {
		if axis >= a.Dims() {
			return nil, axes, retVal, errors.Errorf(errors.DimMismatch, axis, a.Dims())
		}
	}

	switch {
	case toReuse:
		// reshape is already handled by handleFuncOpt
		return ctx, axes, retVal, err
	case safe:
		//	retVal = a.Alike(WithShape(newShape...))
		return ctx, axes, retVal, nil
	case !safe:
		return ctx, axes, retVal, errors.New("Reduce only supports safe operations")
		// case !retVal.IsNativelyAccessible():
		// this case is unreachable because handleFuncOpts already checks this
		// 	return nil, retVal, errors.Errorf(errors.InaccessibleData, retVal)
	}
	panic("Unreachable")
}

func PrepDataVV[DTin, DTout any](a, b tensor.Basic[DTin], reuse tensor.Basic[DTout]) (ait, bit, iit Iterator, useIter, swap bool, err error) {
	useIter = a.RequiresIterator() ||
		b.RequiresIterator() ||
		reuse.RequiresIterator() ||
		!a.DataOrder().HasSameOrder(b.DataOrder())
	if useIter {
		ait = a.Iterator()
		bit = b.Iterator()
		iit = reuse.Iterator()
	}
	return
}

func PrepDataUnary[DT any](a, reuse tensor.Basic[DT]) (ait, rit Iterator, useIter bool, err error) {
	useIter = a.RequiresIterator() || reuse.RequiresIterator()
	if useIter {
		ait = a.Iterator()
		rit = reuse.Iterator()
	}
	return
}

func prepDataVS[DTin, DTout any](a tensor.Basic[DTin], b DTin, reuse tensor.Basic[DTout]) (ait, iit Iterator, useIter, swap bool, err error) {
	useIter = a.RequiresIterator()
	if useIter {
		ait = a.Iterator()
		iit = reuse.Iterator()
	}
	return
}

func prepDataSV[DTin, DTout any](a DTin, b tensor.Basic[DTin], reuse tensor.Basic[DTout]) (bit, iit Iterator, useIter, swap bool, err error) {
	useIter = b.RequiresIterator()
	if useIter {
		bit = b.Iterator()
		iit = reuse.Iterator()
	}
	return
}
