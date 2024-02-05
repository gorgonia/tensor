package tensor

import (
	"gorgonia.org/dtype"
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

func checkCompatibleShape(expected shapes.Shape, others ...shapes.Shape) error {
	expLen := expected.TotalSize()
	for _, s := range others {
		if s.TotalSize() != expLen {
			return errors.Errorf(errors.ShapeMismatch, expected, s)
		}
	}
	return nil

}

// defaultCmpFuncOpt handles the FuncOpts of a comparison function. It adds a `As(dtype.Bool)` in the head of the func opts as by default
// comparison operations return bools.
func defaultCmpFuncOpt(opts []FuncOpt) []FuncOpt {
	opts = append(opts, nil)
	copy(opts[1:], opts[0:])
	opts[0] = As(dtype.Bool) // default
	return opts
}

func prepArith[DT any](a, b Basic[DT], opts ...FuncOpt) (e Engine, newAPA, newAPB *AP, retVal Basic[DT], fo Option, err error) {
	e = getEngine(a, b)
	aShp := a.Shape()
	bShp := b.Shape()
	cShp := getLargestShape(aShp, bShp)

	var prepper FuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(FuncOptHandler[DT]); !ok {
		return nil, nil, nil, nil, fo, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn(1))
	}

	if retVal, fo, err = prepper.HandleFuncOpts(a, cShp, opts...); err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn(1))
	}

	newAPA = a.Info()
	newAPB = b.Info()

	// fast path
	if !fo.Broadcast || aShp.TotalSize() == bShp.TotalSize() {
		// no broadcasting necessary
		fo.Broadcast = false
		return
	}

	newAPA, newAPB = CalcBroadcastShapes(newAPA, newAPB)
	if err = CheckBroadcastable(newAPA.Shape(), newAPB.Shape()); err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}
	return

}

func prepCmp[DT any](a, b Basic[DT], opts ...FuncOpt) (e Engine, newAPA, newAPB *AP, retVal DescWithStorage, fo Option, err error) {
	e = getEngine(a, b)
	aShp := a.Shape()
	bShp := b.Shape()
	cShp := getLargestShape(aShp, bShp)

	var prepper DescFuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(DescFuncOptHandler[DT]); !ok {
		return nil, nil, nil, nil, fo, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn(1))
	}

	opts = defaultCmpFuncOpt(opts)
	if retVal, fo, err = prepper.HandleFuncOptsDesc(a, cShp, opts...); err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn(1))
	}

	newAPA = a.Info()
	newAPB = b.Info()

	// fast path
	if !fo.Broadcast || aShp.TotalSize() == bShp.TotalSize() {
		// no broadcasting necessary
		fo.Broadcast = false
		return
	}

	newAPA, newAPB = CalcBroadcastShapes(newAPA, newAPB)
	if err = CheckBroadcastable(newAPA.Shape(), newAPB.Shape()); err != nil {
		return nil, nil, nil, nil, fo, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn(1))
	}
	return

}
