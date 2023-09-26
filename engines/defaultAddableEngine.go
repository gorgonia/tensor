package stdeng

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

type AddableEng[DT Addable, T tensor.Basic[DT]] struct {
	StdEng[DT, T]
	compAddableEng[DT, T]
}

// compAddableEng is a compositional AddableEngine, which can be used to compose together things. It doesn't implement Engine.
type compAddableEng[DT Addable, T tensor.Basic[DT]] struct{}

func (e AddableEng[DT, T]) BasicEng() Engine { return AddableEng[DT, tensor.Basic[DT]]{} }

func (e compAddableEng[DT, T]) StdBinOp(ctx context.Context, a, b, retVal T, toIncr bool, op Op[DT]) (err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return
	}
	var ait, bit, iit Iterator
	var useIter bool
	if ait, bit, iit, useIter, _, err = PrepDataVV[DT, DT](a, b, retVal); err != nil {
		return errors.Wrap(err, "Unable to prepare iterators for Add")
	}

	if useIter {
		switch {
		case toIncr:
			return internal.HandleNoOp(op.VVIncrIter(a.Data(), b.Data(), retVal.Data(), ait, bit, iit))
		default:
			return internal.HandleNoOp(op.VVIter(a.Data(), b.Data(), retVal.Data(), ait, bit, iit))

		}
	}

	switch {
	case toIncr:
		op.VVIncr(a.Data(), b.Data(), retVal.Data())
		return nil
	default:
		op.VV(a.Data(), b.Data(), retVal.Data())
		return nil
	}
}

func (e compAddableEng[DT, T]) StdBinOpScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool, op Op[DT]) (err error) {
	var ait, iit Iterator
	var useIter bool
	prep := prepDataVS[DT, DT]
	if scalarOnLeft {
		prep = prepDataVS[DT, DT]
	}
	if ait, iit, useIter, _, err = prep(t, s, retVal); err != nil {
		return errors.Wrap(err, "Unable to prepare iterators for Add")
	}

	if useIter {
		switch {
		case !scalarOnLeft && toIncr:
			return internal.HandleNoOp(op.VSIncrIter(t.Data(), s, retVal.Data(), ait, iit))
		case scalarOnLeft && toIncr:
			return internal.HandleNoOp(op.SVIncrIter(s, t.Data(), retVal.Data(), ait, iit))
		case scalarOnLeft && !toIncr:
			return internal.HandleNoOp(op.SVIter(s, t.Data(), retVal.Data(), ait, iit))
		default:
			return internal.HandleNoOp(op.VSIter(t.Data(), s, retVal.Data(), ait, iit))
		}
	}

	switch {
	case !scalarOnLeft && toIncr:
		op.VSIncr(t.Data(), s, retVal.Data())
	case scalarOnLeft && toIncr:
		op.SVIncr(s, t.Data(), retVal.Data())
	case scalarOnLeft && !toIncr:
		op.SV(s, t.Data(), retVal.Data())
	default:
		op.VS(t.Data(), s, retVal.Data())
	}
	return nil
}

func (e compAddableEng[DT, T]) Add(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, toIncr, addOp[DT]())
}

func (e compAddableEng[DT, T]) AddScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, toIncr, addOp[DT]())
}

func (e compAddableEng[DT, T]) Trace(ctx context.Context, t T) (retVal DT, err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}

	if !tensor.IsMatrix(t) {
		return retVal, errors.New("Trace is only defined on matrices")
	}

	rstride := t.Strides()[0]
	cstride := t.Strides()[1]

	r := t.Shape()[0]
	c := t.Shape()[1]

	m := internal.Min(r, c)
	stride := rstride + cstride

	data := t.Data()
	var trace DT
	for i := 0; i < m; i++ {
		trace += data[i*stride]
	}
	retVal = trace

	return retVal, nil
}
