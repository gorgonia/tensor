package dense

import (
	"github.com/chewxy/inigo/values/tensor"
	"github.com/chewxy/inigo/values/tensor/internal/errors"
	"github.com/chewxy/inigo/values/tensor/internal/specialized"
)

func (t *Dense[DT]) ByIndices(indices tensor.Basic[int], axis int, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	methodName := ".ByIndices()"
	if err = check(checkFlags(t.e, t), checkValidAxis(axis, t), checkIsVector(indices)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, methodName)
	}

	// if t is a scalar, then use slice
	if t.Shape().IsScalarEquiv() {
		slices := make([]SliceRange, t.Shape().Dims())
		slices[axis] = SR(indices.Data()[0])
		return t.Slice(slices...)
	}

	expShape := t.Shape().Clone()
	expShape[axis] = indices.Shape().TotalSize()

	h, ok := t.e.(specialized.FuncOptHandler[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, h, methodName)
	}

	retVal, fo, err := h.HandleFuncOptsSpecialized(t, expShape, opts...)
	if err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, methodName)
	}

	e, ok := t.e.(tensor.FancyIndexer[DT, *Dense[DT]])
	if !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, methodName)
	}

	if err = e.SelectByIndices(fo.Ctx, t, indices, axis, retVal); err != nil {
		return nil, err
	}
	return retVal, nil
}
