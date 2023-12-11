package tensor

import (
	"log"

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
		log.Printf("engine %T does not suport %T", e, bla)
		return nil, errors.Errorf(errors.EngineSupport, e, bla, errors.ThisFn())
	}
	// make retVal
	var fo Option
	expShape := elimInnermostOutermost(a.Shape(), b.Shape())
	if retVal, fo, err = handleFuncOpt[DT](e, a, expShape, opts...); err != nil {
		return nil, err
	}
	var incr []DT
	if fo.Incr {
		incr = make([]DT, len(retVal.Data()))
		copy(incr, retVal.Data())
	}
	err = bla.MatMul(fo.Ctx, a, b, retVal, incr)
	return retVal, err
}

func Add[DT Num](a, b Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
	e := getEngine(a, b).BasicEng()
	log.Printf("type of engine %T", e)

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

	// TODO checks

	/*
		switch a := a.(type) {
		case DenseTensor[DT]:
			switch b := b.(type) {
			case DenseTensor[DT]:
				e := getEngine(a, b)
				if e == nil {
					return nil, errors.Errorf("No engines found")
				}
				e = e.BasicEng()
				foh, ok := e.(FuncOptHandler[DT])
				if !ok {
					return nil, errors.Errorf("Engine %T cannot handle func opts", e)
				}
				var fo Option
				if retVal, fo, err = foh.HandleFuncOpts(a, a.Shape(), opts...); err != nil {
					return nil, err
				}

				adder, ok := e.(Adder[DT, Basic[DT]])
				if !ok {
					// error
				}
				ctx := fo.Ctx
				toIncr := fo.Incr
				err = adder.Add(ctx, a, b, retVal, toIncr)
				return retVal, err
				// use dense.Add()
			case SparseTensor[DT]:
				// engine.Add(a.data(), b.data(), ait, bit)
			case Scalar[DT]:
				// a.Engine().AddScalar(a, b.V)
			default:
			}
		case SparseTensor[DT]:
			switch b := b.(type) {
			case DenseTensor[DT]:
				_ = a
				_ = b
			case SparseTensor[DT]:
			case Scalar[DT]:
			default:
			}

		case Scalar[DT]:
			switch b := b.(type) {
			case DenseTensor[DT]:
				_ = a
				_ = b
			case SparseTensor[DT]:
			case Scalar[DT]:
			default:
			}
		default:
			return nil, errors.Errorf("Unhandled tensor type %T", a)
		}
		panic("Unreachable")
	*/
}
