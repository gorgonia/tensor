package stdeng

import (
	"context"

	"gorgonia.org/dtype"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
)

// var _ Comparer[int, *Dense[int]] = StdOrderedNumEngine[int, *Dense[int]]{}

type StdOrderedNumEngine[DT OrderedNum, T tensor.Basic[DT]] struct {
	StdNumEngine[DT, T]
	compComparableEng[DT, T]
}

// Workhorse returns the engine that will actually do all the work (in this case, itself).
func (e StdOrderedNumEngine[DT, T]) Workhorse() Engine { return e }

func (e StdOrderedNumEngine[DT, T]) BasicEng() Engine {
	return StdOrderedNumEngine[DT, tensor.Basic[DT]]{}
}

func (e StdOrderedNumEngine[DT, T]) cmpOp(ctx context.Context, a, b T, retVal tensor.Basic[DT], op Op[DT]) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}
	var ait, bit, iit Iterator
	var useIter bool
	if ait, bit, iit, useIter, _, err = PrepDataVV[DT](a, b, retVal); err != nil {
		return errors.Wrapf(err, "Unable to prepare iterators for %v", errors.ThisFn(2))
	}
	if useIter {
		return op.VVIter(a.Data(), b.Data(), retVal.Data(), ait, bit, iit)
	}

	op.VV(a.Data(), b.Data(), retVal.Data())
	return nil
}

func (e StdOrderedNumEngine[DT, T]) cmpOpScalar(ctx context.Context, a T, b DT, retVal tensor.Basic[DT], scalarOnLeft bool, op Op[DT]) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}

	var tit, iit Iterator
	var useIter bool
	if scalarOnLeft {
		tit, iit, useIter, _, err = prepDataSV[DT](b, a, retVal)
	} else {
		tit, iit, useIter, _, err = prepDataVS[DT](a, b, retVal)
	}
	if err != nil {
		return errors.Wrapf(err, "Unable to prepare iterators for %v", errors.ThisFn(2))
	}

	switch {
	case useIter && scalarOnLeft:
		return op.SVIter(b, a.Data(), retVal.Data(), tit, iit)
	case useIter && !scalarOnLeft:
		return op.VSIter(a.Data(), b, retVal.Data(), tit, iit)
	case !useIter && scalarOnLeft:
		op.SV(b, a.Data(), retVal.Data())
	default:
		op.VS(a.Data(), b, retVal.Data())
	}
	return nil
}

/* ARG CMP */

// flatFn is a function that performs argmax/min on a slice.
// iterFn is a function that performs argmax/min on a slice with an iterator and a last size value.
func (e StdOrderedNumEngine[DT, T]) argCmp(ctx context.Context, a T, axis int, name string, flatFn func([]DT) int, iterFn func([]DT, Iterator, int) ([]int, error)) (retVal tensor.Basic[int], err error) {
	t := any(a).(tensor.Tensor[DT, T])
	if err = internal.HandleCtx(ctx); err != nil {
		return nil, err
	}
	if axis >= t.Dims() {
		return nil, errors.Errorf(errors.DimMismatch, t.Dims(), axis)
	}

	if axis == AllAxes {
		index := flatFn(t.Data())
		_ = index // TODO
		panic("TODO: flat index")
	}

	// ARGMAX ALONG AXIS

	var indices []int
	axes := make([]int, len(t.Shape()))
	for i := range t.Shape() {
		switch {
		case i < axis:
			axes[i] = i
		case i == axis:
			axes[len(axes)-1] = i
		case i > axis:
			axes[i-1] = i
		}
	}
	t2, err := t.T(axes...)
	if err = internal.HandleNoOp(err); err != nil {
		return nil, err
	}
	t2Shp := t2.Shape()
	it := t2.Iterator()
	if indices, err = iterFn(t.Data(), it, t2Shp[len(t2Shp)-1]); err != nil {
		return nil, errors.Wrapf(err, "Failed to perform %sIter", name)
	}
	newShape := t2Shp[:len(t2Shp)-1]

	if aat, ok := any(t).(tensor.AlikeAsTyper); ok {
		retVal = aat.AlikeAsType(dtype.Int, WithShape(newShape.Clone()...), WithBacking(indices)).(tensor.Basic[int])
		return
	}
	return nil, errors.Errorf("tensor is not an AlikeAsTyper")
}
func (e StdOrderedNumEngine[DT, T]) Argmax(ctx context.Context, t T, axis int) (retVal tensor.Basic[int], err error) {
	return e.argCmp(ctx, t, axis, "Argmax", execution.Argmax[DT], execution.ArgmaxIter[DT])
}

func (e StdOrderedNumEngine[DT, T]) Argmin(ctx context.Context, t T, axis int) (retVal tensor.Basic[int], err error) {
	return e.argCmp(ctx, t, axis, "Argmin", execution.Argmin[DT], execution.ArgminIter[DT])
}

/* MIN/MAX */

func (e StdOrderedNumEngine[DT, T]) MinBetween(ctx context.Context, a, b, retVal T) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, false, minOp[DT]())
}

func (e StdOrderedNumEngine[DT, T]) MinBetweenScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, false, minOp[DT]())
}

func (e StdOrderedNumEngine[DT, T]) MaxBetween(ctx context.Context, a, b, retVal T) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, false, maxOp[DT]())
}

func (e StdOrderedNumEngine[DT, T]) MaxBetweenScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, false, maxOp[DT]())
}
