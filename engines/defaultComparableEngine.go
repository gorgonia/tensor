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

// BasicEng turns an engine that has methods that take a specialized T into one that takes tensor.Basic[DT] as inputs.
func (e ComparableEng[DT, T]) BasicEng() Engine {
	return ComparableEng[DT, tensor.Basic[DT]]{}
}

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

func (e compComparableEng[DT, T]) CmpOp(ctx context.Context, a, b T, retVal tensor.Basic[bool], op CmpBinOp[DT]) (err error) {
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

func (e compComparableEng[DT, T]) CmpOpScalar(ctx context.Context, a T, b DT, retVal tensor.Basic[bool], scalarOnLeft bool, op CmpBinOp[DT]) (err error) {
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