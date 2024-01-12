package dense

import (
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

// defaultCmpFuncOpt handles the FuncOpts of a comparison function. It adds a `As(dtype.Bool)` in the head of the func opts as by default
// comparison operations return bools.
func defaultCmpFuncOpt(opts []FuncOpt) []FuncOpt {
	opts = append(opts, nil)
	copy(opts[1:], opts[0:])
	opts[0] = As(dtype.Bool) // default
	return opts
}

func handleFuncOpts[DT any, T tensor.Tensor[DT, T]](e Engine, t T, expShape shapes.Shape, opts ...FuncOpt) (retVal T, fo Option, err error) {
	switch e := e.(type) {
	case tensor.SpecializedFuncOptHandler[DT, T]:
		return e.HandleFuncOptsSpecialized(t, expShape, opts...)
	case tensor.FuncOptHandler[DT]:
		var ret tensor.Basic[DT]
		ret, fo, err = e.HandleFuncOpts(t, expShape, opts...)
		if err != nil {
			return retVal, fo, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
		}
		var ok bool
		if retVal, ok = ret.(T); !ok {
			return retVal, fo, errors.Errorf("Expected retVal type to be %T", retVal)
		}
		return
	case tensor.DescFuncOptHandler[DT]:
		var ret tensor.DescWithStorage
		ret, fo, err = e.HandleFuncOptsDesc(t, expShape, opts...)
		if err != nil {
			return retVal, fo, err
		}
		var ok bool
		if retVal, ok = ret.(T); !ok {
			return retVal, fo, errors.Errorf("Expected retVal type to be %T", retVal)
		}
		return
	}
	return retVal, fo, errors.Errorf(errors.EngineSupport, e, e, errors.ThisFn())
}
