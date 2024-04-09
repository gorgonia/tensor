package dense

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

func (t *Dense[DT]) Argmax(axis int, opts ...FuncOpt) (retVal *Dense[int], err error) {
	fo := ParseFuncOpts(opts...)
	ctx := fo.Ctx

	var e tensor.Argmethoder[DT, *Dense[DT]]
	e, ok := t.e.(tensor.Argmethoder[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}
	ret, err := e.Argmax(ctx, t, axis)
	if err != nil {
		return nil, err
	}
	return ret.(*Dense[int]), nil
}

func (t *Dense[DT]) Argmin(axis int, opts ...FuncOpt) (retVal *Dense[int], err error) {
	fo := ParseFuncOpts(opts...)
	ctx := fo.Ctx

	var e tensor.Argmethoder[DT, *Dense[DT]]
	e, ok := t.e.(tensor.Argmethoder[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}
	ret, err := e.Argmin(ctx, t, axis)
	if err != nil {
		return nil, err
	}
	return ret.(*Dense[int]), nil
}
