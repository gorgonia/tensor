package tensor

import "github.com/chewxy/inigo/values/tensor/internal/errors"

func getEngine(ts ...Engineer) Engine {
	// TODO: get highest capability engine
	for _, t := range ts {
		if e := t.Engine(); e != nil {
			return e
		}
	}
	return nil
}

func Add[DT Num](a, b Basic[DT], opts ...FuncOpt) (retVal Basic[DT], err error) {
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
}
