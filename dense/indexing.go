package dense

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/specialized"
)

func (t *Dense[DT]) ByIndices(indices tensor.Basic[int], axis int, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t), checkValidAxis(axis, t), checkIsVector(indices)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
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
		return nil, errors.Errorf(errors.EngineSupport, t.e, h, errors.ThisFn())
	}

	retVal, fo, err := h.HandleFuncOptsSpecialized(t, expShape, opts...)
	if err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}

	var e tensor.FancyIndexer[DT, *Dense[DT]]
	if e, ok = t.e.(tensor.FancyIndexer[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, e, errors.ThisFn())
	}

	if err = e.SelectByIndices(fo.Ctx, t, indices, axis, retVal); err != nil {
		return nil, err
	}
	return retVal, nil
}

func (t *Dense[DT]) ByIndexesB(indices tensor.Basic[int], outGrad *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	panic("NYI")
}

func (t *Dense[DT]) Scatter(indices Densor[int]) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.Engine(), t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	//TODO: support funcops
	indicesT := indices.GetDense()
	maxT, err := Max[int](indicesT)
	if err != nil {
		return nil, err
	}
	max := maxT.ScalarValue()

	expShape := indicesT.Shape().Clone()
	expShape[len(expShape)-1] = max + 1
	retVal = New[DT](WithShape(expShape...), WithEngine(t.Engine()))

	var sc tensor.Scatterer[DT]
	var ok bool
	if sc, ok = t.e.(tensor.Scatterer[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, sc, errors.ThisFn())
	}
	err = sc.Scatter(context.Background(), t, indicesT, retVal)
	return
}
