package tensor

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal/errors"
)

func getEngine(ts ...Engineer) Engine {
	// TODO: get highest capability engine
	for _, t := range ts {
		if e := t.Engine(); e != nil {
			return e
		}
	}
	return nil
}

func handleFuncOpt[DT any](e Engine, t Basic[DT], expShape shapes.Shape, opts ...FuncOpt) (retVal Basic[DT], fo Option, err error) {
	h := e.(FuncOptHandler[DT])
	return h.HandleFuncOpts(t, expShape, opts...)
}
func elimInnermostOutermost(a, b shapes.Shape) shapes.Shape {
	a2 := a.Clone()
	return append(a2[:len(a)-1], b[1:]...)
}

func getLargestShape(ss ...shapes.Shape) shapes.Shape {
	var max shapes.Shape
	var maxSize int
	for _, s := range ss {
		sz := s.TotalSize()
		if sz > maxSize {
			max = s
			maxSize = sz
		}
	}
	return max
}

func MatMul[DT Num](a, b Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
	e := getEngine(a, b).BasicEng()

	// TODO: checks

	var bla BLA[DT, Basic[DT]]
	var ok bool
	if bla, ok = e.(BLA[DT, Basic[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, bla, errors.ThisFn())
	}
	// make retVal
	var fo Option
	expShape := elimInnermostOutermost(a.Shape(), b.Shape())
	if retVal, fo, err = handleFuncOpt[DT](e, a, expShape, opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	var incr []DT
	if fo.Incr {
		incr = make([]DT, len(retVal.Data()))
		copy(incr, retVal.Data()) // may panic if engine is a non-local engine (e.g. CUDA).
	}
	err = bla.MatMul(fo.Ctx, a, b, retVal, incr)
	return retVal, err
}

func Add[DT Num](a, b Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
	e := getEngine(a, b).BasicEng()

	var adder Adder[DT, Basic[DT]]
	var ok bool
	if adder, ok = e.(Adder[DT, Basic[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, adder, errors.ThisFn())
	}

	var fo Option
	ashp := a.Shape()
	bshp := b.Shape()
	expShape := getLargestShape(ashp, bshp)
	if retVal, fo, err = handleFuncOpt[DT](e, a, expShape, opts...); err != nil {
		return nil, err
	}

	switch {
	case ashp.IsScalarEquiv() && bshp.IsScalarEquiv():
		err = adder.Add(fo.Ctx, a, b, retVal, fo.Incr)
	case ashp.IsScalarEquiv() && !bshp.IsScalarEquiv():
		err = adder.AddScalar(fo.Ctx, b, a.Data()[0], retVal, true, fo.Incr)
	case !ashp.IsScalarEquiv() && bshp.IsScalarEquiv():
		err = adder.AddScalar(fo.Ctx, a, b.Data()[0], retVal, false, fo.Incr)
	default:
		err = adder.Add(fo.Ctx, a, b, retVal, fo.Incr)
	}

	return retVal, err
}

func Abs[DT Num](a Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
	e := getEngine(a).BasicEng()
	var abser Abser[DT, Basic[DT]]
	var ok bool
	if abser, ok = e.(Abser[DT, Basic[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, abser, errors.ThisFn())
	}
	var fo Option
	if retVal, fo, err = handleFuncOpt[DT](e, a, a.Shape(), opts...); err != nil {
		return nil, err
	}
	if err = abser.Abs(fo.Ctx, a, retVal); err != nil {
		return nil, err
	}
	return retVal, err
}
