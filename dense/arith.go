// Code generated by genlib3. DO NOT EDIT

package dense

import (
	"context"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
	"log"
)

func (t *Dense[DT]) basicArithPrep(u *Dense[DT], opts ...FuncOpt) (e Engine, newAPT, newAPU *tensor.AP, retVal *Dense[DT], fo Option, err error) {
	e = getEngine[DT](t, u)
	if err = check(checkFlags(e, t, u)); err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}
	tShp := t.Shape()
	uShp := u.Shape()
	expShape := largestShape(tShp, uShp)

	retVal, fo, err = handleFuncOpts[DT, *Dense[DT]](e, t, expShape, opts...)
	if err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}

	newAPT = t.Info()
	newAPU = u.Info()

	// fast path
	if !fo.Broadcast || tShp.TotalSize() == uShp.TotalSize() {
		// no broadcasting necessary
		fo.Broadcast = false
		return
	}

	// create autobroadcast shape
	newAPT, newAPU = tensor.CalcBroadcastShapes(newAPT, newAPU)
	if err = tensor.CheckBroadcastable(newAPT.Shape(), newAPU.Shape()); err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}

	return
}

func (t *Dense[DT]) basicArithScalarPrep(s DT, opts ...FuncOpt) (e Engine, retVal *Dense[DT], ctx context.Context, toIncr bool, err error) {
	e = getEngine[DT](t)
	if err = check(checkFlags(e, t)); err != nil {
		return nil, nil, nil, false, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}
	var fo Option
	retVal, fo, err = handleFuncOpts[DT, *Dense[DT]](e, t, t.Shape(), opts...)
	if err != nil {
		return nil, nil, nil, false, err
	}

	toIncr = fo.Incr
	ctx = fo.Ctx
	return
}

// Add performs `t + u`.
func (t *Dense[DT]) Add(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, newAPT, newAPU, retVal, fo, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}
	ctx := fo.Ctx
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	adder, ok := e.(tensor.Adder[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, adder, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = adder.AddBroadcastable(ctx, t, u, retVal, newAPT, newAPU, toIncr)
	default:
		err = adder.Add(ctx, t, u, retVal, toIncr)

	}
	return retVal, err
}

// AddScalar performs `t + s`. If `scalarOnLeft` is true, then it performs `s + t`.
func (t *Dense[DT]) AddScalar(s DT, scalarOnLeft bool, opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithScalarPrep(s, opts...)
	if err != nil {
		return nil, err
	}

	adder, ok := e.(tensor.Adder[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, adder, errors.ThisFn())
	}

	if err = adder.AddScalar(ctx, t, s, retVal, scalarOnLeft, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

// Sub performs `t - u`.
func (t *Dense[DT]) Sub(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, newAPT, newAPU, retVal, fo, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}
	ctx := fo.Ctx
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	suber, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, suber, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = suber.SubBroadcastable(ctx, t, u, retVal, newAPT, newAPU, toIncr)
	default:
		err = suber.Sub(ctx, t, u, retVal, toIncr)

	}
	return retVal, err
}

// SubScalar performs `t - s`. If `scalarOnLeft` is true, then it performs `s - t`.
func (t *Dense[DT]) SubScalar(s DT, scalarOnLeft bool, opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithScalarPrep(s, opts...)
	if err != nil {
		return nil, err
	}

	suber, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, suber, errors.ThisFn())
	}

	if err = suber.SubScalar(ctx, t, s, retVal, scalarOnLeft, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

// Mul performs `t * u`.
func (t *Dense[DT]) Mul(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, newAPT, newAPU, retVal, fo, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}
	ctx := fo.Ctx
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	muler, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		log.Printf("BasicArither? e %v %T", e, e)
		return nil, errors.Errorf(errors.EngineSupport, e, muler, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = muler.MulBroadcastable(ctx, t, u, retVal, newAPT, newAPU, toIncr)
	default:
		err = muler.Mul(ctx, t, u, retVal, toIncr)

	}
	return retVal, err
}

// MulScalar performs `t * s`. If `scalarOnLeft` is true, then it performs `s * t`.
func (t *Dense[DT]) MulScalar(s DT, scalarOnLeft bool, opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithScalarPrep(s, opts...)
	if err != nil {
		return nil, err
	}

	muler, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, muler, errors.ThisFn())
	}

	if err = muler.MulScalar(ctx, t, s, retVal, scalarOnLeft, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

// Div performs `t / u`.
func (t *Dense[DT]) Div(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, newAPT, newAPU, retVal, fo, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}
	ctx := fo.Ctx
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	diver, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, diver, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = diver.DivBroadcastable(ctx, t, u, retVal, newAPT, newAPU, toIncr)
	default:
		err = diver.Div(ctx, t, u, retVal, toIncr)

	}
	return retVal, err
}

// DivScalar performs `t / s`. If `scalarOnLeft` is true, then it performs `s / t`.
func (t *Dense[DT]) DivScalar(s DT, scalarOnLeft bool, opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithScalarPrep(s, opts...)
	if err != nil {
		return nil, err
	}

	diver, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, diver, errors.ThisFn())
	}

	if err = diver.DivScalar(ctx, t, s, retVal, scalarOnLeft, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}
