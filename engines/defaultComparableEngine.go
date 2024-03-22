package stdeng

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

type ComparableEng[DT comparable, T tensor.Basic[DT]] struct {
	StdEng[DT, T]
	compComparableEng[DT, T]
}

// compComparableEng is a compositional ComparableEng. It doesn't fully implement Engine, but rather exists to give ComparableEngine its comparable capabilities
type compComparableEng[DT comparable, T tensor.Basic[DT]] struct{}

// Workhorse returns the engine that will actually do all the work (in this case, itself).
func (e ComparableEng[DT, T]) Workhorse() Engine { return e }

func (e ComparableEng[DT, T]) SliceEq(a, b []DT) bool {
	if internal.SliceEqMeta(a, b) {
		return true
	}

	if len(a) != len(b) {
		return false
	}

	// otherwise we will have to compare it elementwise
	a = a[:len(a)]
	b = b[:len(a)]
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func (e compComparableEng[DT, T]) CmpOp(ctx context.Context, a, b tensor.Basic[DT], retVal tensor.Basic[bool], op CmpBinOp[DT]) (err error) {
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

func (e compComparableEng[DT, T]) CmpOpBC(ctx context.Context, a, b tensor.Basic[DT], retVal tensor.Basic[bool], expAPA, expAPB *tensor.AP, op CmpBinOp[DT]) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return err
	}
	var useIter bool
	if _, _, _, useIter, _, err = PrepDataVV[DT](a, b, retVal); err != nil {
		return errors.Wrapf(err, "Unable to prepare iterators for %v", errors.ThisFn(2))
	}
	if useIter {
		return errors.Errorf(errors.NYIPR, errors.ThisFn(1), "Broadcasting operation for tensors that require use of iterators")
	}
	expShapeA := expAPA.Shape()
	expShapeB := expAPB.Shape()
	expStridesA := expAPA.Strides()
	expStridesB := expAPB.Strides()

	op.VVBC(a.Data(), b.Data(), retVal.Data(), expShapeA, expShapeB, retVal.Shape(), expStridesA, expStridesB)
	return nil
}

func (e compComparableEng[DT, T]) CmpOpScalar(ctx context.Context, a tensor.Basic[DT], b DT, retVal tensor.Basic[bool], scalarOnLeft bool, op CmpBinOp[DT]) (err error) {
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
