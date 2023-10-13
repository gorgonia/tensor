package dense

import (
	"context"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/specialized"
)

func (t *Dense[DT]) Apply(fn any, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	h, ok := t.e.(specialized.FuncOptHandler[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, h, errors.ThisFn())
	}
	var fo Option
	retVal, fo, err = h.HandleFuncOptsSpecialized(t, t.Shape(), opts...)
	if err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if fo.Incr {
		// TODO: error
	}
	ctx := fo.Ctx

	e, ok := t.e.(tensor.Mapper[DT, *Dense[DT]])
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

	var e tensor.Reducer[DT, *Dense[DT]]
	var ok bool
	if e, ok = t.e.(tensor.Reducer[DT, *Dense[DT]]); !ok {
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
		return reuse, nil
	}

	module, err := redFunc2Mod[DT](fn)
	if err != nil {
		return nil, errors.Wrap(err, "Cannot Reduce. Invalid function provided.")
	}

	if err = e.ReduceAlong(ctx, module, defaultVal, t, reuse, axes...); err != nil {
		return nil, errors.Wrap(err, "Cannot Reduce. ReduceAlong caused an error.")
	}
	return reuse, nil
}

func (t *Dense[DT]) Scan(fn func(a, b DT) DT, axis int, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	h, ok := t.e.(specialized.FuncOptHandler[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, h, errors.ThisFn())
	}
	var fo Option
	if retVal, fo, err = h.HandleFuncOptsSpecialized(t, t.Shape(), opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	ctx := fo.Ctx

	var e tensor.Scanner[DT, *Dense[DT]]
	if e, ok = t.e.(tensor.Scanner[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}
	err = e.Scan(ctx, fn, t, axis, retVal)
	return retVal, err
}

func (t *Dense[DT]) Dot(reductionFn, elwiseFn func(DT, DT) DT, other *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	if err := check(checkFlags(t.e, t), checkInnerProdDims(t, other)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	e, ok := t.e.(tensor.DotIterer[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}

	h, ok := t.e.(specialized.FuncOptHandler[DT, *Dense[DT]])
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

	var txer tensor.Transposer[DT, *Dense[DT]]
	var ok bool
	if txer, ok = t.e.(tensor.Transposer[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, txer, errors.ThisFn())
	}

	if err = txer.Transpose(context.Background(), u, expStrides); err != nil {
		return nil, err
	}
	u.AP.RecalcStrides()
	return u, nil
}

func (t *Dense[DT]) Repeat(axis int, repeats ...int) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t), checkRepeatValidAxis(axis, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	e, ok := t.e.(tensor.Repeater[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}
	ctx, retVal, newAxis, size, newRepeats, err := e.PrepRepeat(t, axis, repeats)
	if err != nil {
		return nil, err
	}
	err = e.Repeat(ctx, t, retVal, newAxis, size, newRepeats)
	return retVal, err
}
