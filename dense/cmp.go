package dense

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/specialized"
	"gorgonia.org/dtype"
)

func (t *Dense[DT]) Lt(u *Dense[DT], opts ...FuncOpt) (retVal DescWithStorage, err error) {
	e := getEngine[DT](t, u)
	if err = check(checkFlags(e, t, u), checkEqShape(t.Shape(), u.Shape())); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	var prepper specialized.FuncOptHandler[DT, *Dense[DT]]
	var ok bool
	if prepper, ok = e.(specialized.FuncOptHandler[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}

	var fo Option
	if retVal, fo, err = prepper.HandleFuncOptsSpecialized(t, t.Shape(), opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if fo.Incr {
		return nil, errors.Errorf("Unable to Incr for Lt")
	}

	asBool := fo.AsType == dtype.Bool
	ctx := fo.Ctx

	var cmper tensor.Comparer[DT, *Dense[DT]]
	if cmper, ok = e.(tensor.Comparer[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, cmper, errors.ThisFn())
	}
	if err = cmper.Lt(ctx, t, u, retVal, asBool); err != nil {
		return nil, err
	}
	return retVal, nil
}
