// Code generated by genlib3. DO NOT EDIT

package stdeng

import (
	"context"

	"gorgonia.org/tensor"
)

// Lt performs `a < b`, with a bool tensor as the return value.
func (e StdOrderedNumEngine[DT, T]) Lt(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error) {
	op, cmpOp := ltOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), cmpOp)
	}
	return e.cmpOp(ctx, a, b, retVal.(tensor.Basic[DT]), op)
}

// LtScalar performs `vec < scalar` or `scalar < vec`, with a bool tensor as the return value. The `scalarOnLeft` parameter indicates
// if the scalar value is on the left of the bin op
func (e StdOrderedNumEngine[DT, T]) LtScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op, cmpOp := ltOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, cmpOp)
	}
	return e.cmpOpScalar(ctx, a, b, retVal.(tensor.Basic[DT]), scalarOnLeft, op)
}

// Lte performs `a <= b`, with a bool tensor as the return value.
func (e StdOrderedNumEngine[DT, T]) Lte(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error) {
	op, cmpOp := lteOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), cmpOp)
	}
	return e.cmpOp(ctx, a, b, retVal.(tensor.Basic[DT]), op)
}

// LteScalar performs `vec <= scalar` or `scalar <= vec`, with a bool tensor as the return value. The `scalarOnLeft` parameter indicates
// if the scalar value is on the left of the bin op
func (e StdOrderedNumEngine[DT, T]) LteScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op, cmpOp := lteOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, cmpOp)
	}
	return e.cmpOpScalar(ctx, a, b, retVal.(tensor.Basic[DT]), scalarOnLeft, op)
}

// Gt performs `a > b`, with a bool tensor as the return value.
func (e StdOrderedNumEngine[DT, T]) Gt(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error) {
	op, cmpOp := gtOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), cmpOp)
	}
	return e.cmpOp(ctx, a, b, retVal.(tensor.Basic[DT]), op)
}

// GtScalar performs `vec > scalar` or `scalar > vec`, with a bool tensor as the return value. The `scalarOnLeft` parameter indicates
// if the scalar value is on the left of the bin op
func (e StdOrderedNumEngine[DT, T]) GtScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op, cmpOp := gtOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, cmpOp)
	}
	return e.cmpOpScalar(ctx, a, b, retVal.(tensor.Basic[DT]), scalarOnLeft, op)
}

// Gte performs `a >= b`, with a bool tensor as the return value.
func (e StdOrderedNumEngine[DT, T]) Gte(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error) {
	op, cmpOp := gteOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), cmpOp)
	}
	return e.cmpOp(ctx, a, b, retVal.(tensor.Basic[DT]), op)
}

// GteScalar performs `vec >= scalar` or `scalar >= vec`, with a bool tensor as the return value. The `scalarOnLeft` parameter indicates
// if the scalar value is on the left of the bin op
func (e StdOrderedNumEngine[DT, T]) GteScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op, cmpOp := gteOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, cmpOp)
	}
	return e.cmpOpScalar(ctx, a, b, retVal.(tensor.Basic[DT]), scalarOnLeft, op)
}

// ElEq performs `a == b`, with a bool tensor as the return value.
func (e StdOrderedNumEngine[DT, T]) ElEq(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error) {
	op, cmpOp := eleqOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), cmpOp)
	}
	return e.cmpOp(ctx, a, b, retVal.(tensor.Basic[DT]), op)
}

// ElEqScalar performs `vec == scalar` or `scalar == vec`, with a bool tensor as the return value. The `scalarOnLeft` parameter indicates
// if the scalar value is on the left of the bin op
func (e StdOrderedNumEngine[DT, T]) ElEqScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op, cmpOp := eleqOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, cmpOp)
	}
	return e.cmpOpScalar(ctx, a, b, retVal.(tensor.Basic[DT]), scalarOnLeft, op)
}

// Ne performs `a != b`, with a bool tensor as the return value.
func (e StdOrderedNumEngine[DT, T]) Ne(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error) {
	op, cmpOp := neOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOp(ctx, a, b, retVal.(tensor.Basic[bool]), cmpOp)
	}
	return e.cmpOp(ctx, a, b, retVal.(tensor.Basic[DT]), op)
}

// NeScalar performs `vec != scalar` or `scalar != vec`, with a bool tensor as the return value. The `scalarOnLeft` parameter indicates
// if the scalar value is on the left of the bin op
func (e StdOrderedNumEngine[DT, T]) NeScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, asSameDT bool) (err error) {
	op, cmpOp := neOpOrderedNum[DT]()
	if !asSameDT {
		return e.CmpOpScalar(ctx, a, b, retVal.(tensor.Basic[bool]), scalarOnLeft, cmpOp)
	}
	return e.cmpOpScalar(ctx, a, b, retVal.(tensor.Basic[DT]), scalarOnLeft, op)
}
