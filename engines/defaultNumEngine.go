package stdeng

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

type StdNumEngine[DT Num, T tensor.Basic[DT]] struct {
	StdEng[DT, T]
	compAddableEng[DT, T]
	compComparableEng[DT, T]
}

// BasicEng turns an engine that has methods that take a specialized T into one that takes tensor.Basic[DT] as inputs.
func (e StdNumEngine[DT, T]) BasicEng() Engine { return StdNumEngine[DT, tensor.Basic[DT]]{} }

// Workhorse returns the engine that will actually do all the work (in this case, itself).
func (e StdNumEngine[DT, T]) Workhorse() Engine { return e }

func (e StdNumEngine[DT, T]) SliceEq(a, b []DT) bool {
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

func (e StdNumEngine[DT, T]) Sub(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, toIncr, subOp[DT]())
}

func (e StdNumEngine[DT, T]) SubScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, toIncr, subOp[DT]())
}

func (e StdNumEngine[DT, T]) SubBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	return e.StdBinOpBC(ctx, a, b, retVal, expAPA, expAPB, toIncr, subOp[DT]())
}

func (e StdNumEngine[DT, T]) Mul(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, toIncr, mulOp[DT]())
}

func (e StdNumEngine[DT, T]) MulScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, toIncr, mulOp[DT]())
}

func (e StdNumEngine[DT, T]) MulBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	return e.StdBinOpBC(ctx, a, b, retVal, expAPA, expAPB, toIncr, mulOp[DT]())
}

func (e StdNumEngine[DT, T]) Div(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, toIncr, divOp[DT]())
}

func (e StdNumEngine[DT, T]) DivScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, toIncr, divOp[DT]())
}

func (e StdNumEngine[DT, T]) DivBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *tensor.AP, toIncr bool) (err error) {
	return e.StdBinOpBC(ctx, a, b, retVal, expAPA, expAPB, toIncr, divOp[DT]())
}

func (e StdNumEngine[DT, T]) Inner(ctx context.Context, a, b T) (retVal DT, err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}

	A := a.Data()
	B := b.Data()
	if len(A) != len(B) {
		return retVal, errors.Errorf(errors.ShapeMismatch, a.Shape(), b.Shape())
	}

	ret := make([]DT, len(A))
	for i, v := range A {
		ret[i] = v * B[i]
	}

	for i := range ret {
		retVal += ret[i]
	}
	return
}
