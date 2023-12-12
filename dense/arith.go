package dense

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

// arith.go contains all the arithmetic methods for `*Dense[DT]`

func (t *Dense[DT]) basicArithPrep(u *Dense[DT], opts ...FuncOpt) (e Engine, retVal *Dense[DT], ctx context.Context, toIncr bool, err error) {
	e = getEngine[DT](t, u)
	if err = check(checkFlags(e, t, u), checkEqShape(t.Shape(), u.Shape())); err != nil {
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

func (t *Dense[DT]) Add(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}

	adder, ok := e.(tensor.Adder[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, adder, errors.ThisFn())
	}

	if err = adder.Add(ctx, t, u, retVal, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

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

func (t *Dense[DT]) Sub(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}

	subber, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, subber, errors.ThisFn())
	}

	if err = subber.Sub(ctx, t, u, retVal, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

func (t *Dense[DT]) Mul(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}

	muler, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, muler, errors.ThisFn())
	}

	if err = muler.Mul(ctx, t, u, retVal, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}

func (t *Dense[DT]) Div(u *Dense[DT], opts ...FuncOpt) (*Dense[DT], error) {
	e, retVal, ctx, toIncr, err := t.basicArithPrep(u, opts...)
	if err != nil {
		return nil, err
	}

	diver, ok := e.(tensor.BasicArither[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, diver, errors.ThisFn())
	}

	if err = diver.Div(ctx, t, u, retVal, toIncr); err != nil {
		return nil, err
	}
	return retVal, nil
}
