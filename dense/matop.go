package dense

import (
	"context"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

// Apply applies the function `fn` to all the elements of the tensor. The function `fn` must be of type `func(DT) DT` or `func(DT) (DT, error)`.
func (t *Dense[DT]) Apply(fn any, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	h, ok := t.e.(tensor.SpecializedFuncOptHandler[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, h, errors.ThisFn())
	}
	var fo Option
	retVal, fo, err = h.HandleFuncOptsSpecialized(t, t.Shape(), opts...)
	if err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if fo.Incr {
		return retVal, errors.Errorf("Cannot apppy then increment a tensor")
	}
	ctx := fo.Ctx

	e, ok := t.e.(tensor.Mapper[DT])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}
	if err = e.Map(ctx, fn, t, retVal); err != nil {
		return nil, err
	}
	return retVal, nil

}

func (t *Dense[DT]) Reduce(fn any, defaultVal DT, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	var e tensor.Reducer[DT]
	var ok bool
	if e, ok = t.e.(tensor.Reducer[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}

	ctx, axes, reuse, err := e.PrepReduce(t, opts...)
	if err != nil {
		return nil, errors.Wrap(err, "PrepReduce")
	}

	if len(axes) == 1 {
		if err = e.Reduce(ctx, fn, t, axes[0], defaultVal, reuse); err != nil {
			return nil, err
		}
		return reuse.(*Dense[DT]), nil
	}

	module, err := redFunc2Mod[DT](fn)
	if err != nil {
		return nil, errors.Wrap(err, "Cannot Reduce. Invalid function provided.")
	}

	if err = e.ReduceAlong(ctx, module, defaultVal, t, reuse, axes...); err != nil {
		return nil, errors.Wrap(err, "Cannot Reduce. ReduceAlong caused an error.")
	}
	return reuse.(*Dense[DT]), nil
}

func (t *Dense[DT]) Scan(fn func(a, b DT) DT, axis int, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	h, ok := t.e.(tensor.SpecializedFuncOptHandler[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, h, errors.ThisFn())
	}
	var fo Option
	if retVal, fo, err = h.HandleFuncOptsSpecialized(t, t.Shape(), opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	ctx := fo.Ctx

	var e tensor.Scanner[DT]
	if e, ok = t.e.(tensor.Scanner[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}
	err = e.Scan(ctx, fn, t, axis, retVal)
	return retVal, err
}

func (t *Dense[DT]) Dot(reductionFn, elwiseFn func(DT, DT) DT, other *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	if err := check(checkFlags(t.e, t), checkInnerProdDims(t, other)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	e, ok := t.e.(tensor.DotIterer[DT])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}

	h, ok := t.e.(tensor.SpecializedFuncOptHandler[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, h, errors.ThisFn())
	}
	expShape := shapes.Shape{t.Shape()[0], other.Shape()[1]}
	retVal, fo, err := h.HandleFuncOptsSpecialized(t, expShape, opts...)
	if err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	ctx := fo.Ctx
	err = e.DotIter(ctx, reductionFn, elwiseFn, t, other, retVal)
	return retVal, err
}

// Transpose transposes the *Dense tensor according to the given axes. The primary difference between
// the .T() method and the .Transpose method is that the .Transpose() method actually performs the memory move, thus it does not return a
// view on memory. The `.T()` method on the other hand returns a view on memory.
func (t *Dense[DT]) Transpose(axes ...int) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	u := t.Clone()

	var transform AP
	if transform, axes, err = u.AP.T(axes...); err != nil {
		switch err := err.(type) {
		case errors.NoOp:
			return t, err
		default:
			return nil, err
		}
	}
	// AP.T() returns a data order with a transposed flag.
	// This is to help with the .T() method.
	// But when we are transposing, we are moving data, so the flag is no longer true
	// thus we have to clear it
	transform.SetDataOrder(transform.DataOrder().ClearTransposed())
	u.AP = transform
	expStrides := transform.Strides()

	var txer tensor.Transposer[DT]
	var ok bool
	if txer, ok = t.e.(tensor.Transposer[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, txer, errors.ThisFn())
	}

	if err = txer.Transpose(context.Background(), u, expStrides); err != nil {
		return nil, err
	}
	u.AP.RecalcStrides()
	return u, nil
}

// Repeat repeats the *Dense tensor along the given axis the given number of times.
// If the axis is -1, it repeats all axes.
// If the axis is 0, it repeats the rows.
// If the axis is 1, it repeats the columns.
// Axis must be within the bounds of the shape of the tensor.
// Repeats must be positive integers.
func (t *Dense[DT]) Repeat(axis int, repeats ...int) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t), checkRepeatValidAxis(axis, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	e, ok := t.e.(tensor.Repeater[DT])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}
	ctx, ret, newAxis, size, newRepeats, err := e.PrepRepeat(t, axis, repeats)
	if err != nil {
		return nil, err
	}
	err = e.Repeat(ctx, t, ret, newAxis, size, newRepeats)
	return ret.(*Dense[DT]), err
}

// Concat concatenates the receiver with the given tensors along the given axis.
// The axis must be within the bounds of the shape of the receiver.
// The shape of the tensors to be concatenated must be the same as the shape of the receiver, except for the axis of concatenation.
func (t *Dense[DT]) Concat(axis int, tensors ...*Dense[DT]) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	e, ok := t.e.(tensor.Concater[DT])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}

	return e.Concat(context.Background(), t, axis, tensors...)
}

// CopyTo copies the underlying data to the destination. The original data is untouched.
// Note: CopyTo doesn't care about the metadata of the destination *Dense[DT].
func (t *Dense[DT]) CopyTo(dest *Dense[DT]) error {
	if dest == t {
		return nil
	}
	if dest.Size() != t.Size() {
		return errors.Errorf(errors.SizeMismatch, t.Size(), dest.Size())
	}
	tf := t.Flags()
	df := dest.Flags()
	//te := t.Engine()
	//de := d.Engine()
	switch {
	case !tf.IsNativelyAccessible() && !df.IsNativelyAccessible():
		// if it's the same engine, then we can just copy
		panic("NYI")
	case !tf.IsNativelyAccessible() && df.IsNativelyAccessible():
		panic("NYI")
	case tf.IsNativelyAccessible() && !df.IsNativelyAccessible():
		panic("NYI")
	default:
		// if both are natively accessible then we can just copy using the built in copy
		copy(dest.Data(), t.Data())
		return nil
	}
	panic("Unreachable")

}
