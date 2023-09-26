package stdeng

import (
	"context"

	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
)

type StdNumEngine[DT Num, T tensor.Basic[DT]] struct {
	StdEng[DT, T]
	compAddableEng[DT, T]
	compComparableEng[DT, T]
}

// BasicEng turns an engine that has methods that take a specialized T into one that takes tensor.Basic[DT] as inputs.
func (e StdNumEngine[DT, T]) BasicEng() Engine { return StdNumEngine[DT, tensor.Basic[DT]]{} }

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

func (e StdNumEngine[DT, T]) Mul(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, toIncr, mulOp[DT]())
}

func (e StdNumEngine[DT, T]) MulScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, toIncr, mulOp[DT]())
}

func (e StdNumEngine[DT, T]) Div(ctx context.Context, a, b, retVal T, toIncr bool) (err error) {
	return e.StdBinOp(ctx, a, b, retVal, toIncr, divOp[DT]())
}

func (e StdNumEngine[DT, T]) DivScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error) {
	return e.StdBinOpScalar(ctx, t, s, retVal, scalarOnLeft, toIncr, divOp[DT]())
}
