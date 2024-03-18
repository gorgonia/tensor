package tensor

import (
	"context"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal/errors"
)

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
	e := GetEngine(a, b).BasicEng()

	// TODO: checks

	var bla BLA[DT]
	var ok bool
	if bla, ok = e.(BLA[DT]); !ok {
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

func Abs[DT Num](a Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
	e := GetEngine(a).BasicEng()
	var abser Abser[DT]
	var ok bool
	if abser, ok = e.(Abser[DT]); !ok {
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

func MatVecMul[DT Num](a, b Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
	e := GetEngine(a, b).BasicEng()
	var mvmer BLA[DT]
	var ok bool
	if mvmer, ok = e.(BLA[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, mvmer, errors.ThisFn())
	}
	var fo Option
	expShape := elimInnermostOutermost(a.Shape(), b.Shape())
	if retVal, fo, err = handleFuncOpt[DT](e, a, expShape, opts...); err != nil {
		return nil, err
	}

	var incr []DT
	if fo.Incr {
		incr = make([]DT, len(retVal.Data()))
		copy(incr, retVal.Data()) // may panic if engine is a non-local engine (e.g. CUDA).
	}

	if err = mvmer.MatVecMul(fo.Ctx, a, b, retVal, incr); err != nil {
		return nil, err
	}
	return retVal, err
}

func Inner[DT Num](a, b Basic[DT]) (retVal DT, err error) {
	e := GetEngine(a, b).BasicEng()
	var innerer InnerProder[DT]
	var ok bool
	if innerer, ok = e.(InnerProder[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, innerer, errors.ThisFn())
	}
	return innerer.Inner(context.Background(), a, b)
}

func Outer[DT Num](a, b Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
	e := GetEngine(a, b).BasicEng()
	var outerer BLA[DT]
	var ok bool
	if outerer, ok = e.(BLA[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, outerer, errors.ThisFn())
	}
	var fo Option
	expShape := shapes.Shape{a.Shape().TotalSize(), b.Shape().TotalSize()}
	if retVal, fo, err = handleFuncOpt[DT](e, a, expShape, opts...); err != nil {
		return nil, err
	}
	if err = outerer.Outer(fo.Ctx, a, b, retVal, nil); err != nil {
		return nil, err
	}
	return retVal, err
}
