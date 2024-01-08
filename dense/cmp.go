package dense

import (
	"gorgonia.org/dtype"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
	"log"
)

func (t *Dense[DT]) Lt(u *Dense[DT], opts ...FuncOpt) (retVal DescWithStorage, err error) {
	e := getEngine[DT](t, u)
	if err = check(checkFlags(e, t, u)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	tShp := t.Shape()
	uShp := u.Shape()
	expShape := largestShape(tShp, uShp)

	var prepper tensor.DescFuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.DescFuncOptHandler[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}

	var fo Option
	if retVal, fo, err = prepper.HandleFuncOptsDesc(t, expShape, opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if fo.Incr {
		return nil, errors.Errorf("Unable to Incr for Lt")
	}

	asBool := fo.AsType == dtype.Bool
	ctx := fo.Ctx

	var cmper tensor.Ord[DT, *Dense[DT]]
	if cmper, ok = e.(tensor.Ord[DT, *Dense[DT]]); !ok {
		log.Printf("ORD FAIL")
		return nil, errors.Errorf(errors.EngineSupport, e, cmper, errors.ThisFn())
	}

	if fo.Broadcast {
		// create Autobroadcast shape
		newAPT, newAPU := tensor.CalcBroadcastShapes(t.Info(), u.Info())
		if err = tensor.CheckBroadcastable(newAPT.Shape(), newAPU.Shape()); err != nil {
			return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
		}

		err = cmper.LtBroadcastable(ctx, t, u, retVal, !asBool, newAPT, newAPU)
		return
	}

	if err = cmper.Lt(ctx, t, u, retVal, !asBool); err != nil {
		return nil, err
	}
	return retVal, nil
}
