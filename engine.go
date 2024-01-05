package tensor

import (
	"context"

	"gorgonia.org/shapes"
)

/* Data Agnostic Execution Engine Methods */

// Transposer is any engine that can perform an unsafe transpose of a tensor.
type Transposer[DT any, T Tensor[DT, T]] interface {
	Transpose(ctx context.Context, t T, expStrides []int) error
}

/*
// Concater is any enegine that can concatenate multiple Tensors together
type Concater[T any] interface {
	Concat(ctx context.Context, t Tensor[T], axis int, others ...Tensor) (Tensor, error)
}

// Stacker is any engine that can stack multiple Tenosrs along an axis
type Stacker interface {
	Stack(ctx context.Context, t Tensor, axis int, others ...Tensor) (Tensor, error)
}

// DenseStacker is any engine that can stack DenseTensors along an axis. This is a specialization of Stacker.
type DenseStacker interface {
	StackDense(ctx context.Context, t DenseTensor, axis int, others ...DenseTensor) (retVal DenseTensor, err error)
}


// Diager is any engine that can return a tensor that only contains the diagonal values of the input
type Diager interface {
	Diag(ctx context.Context, a Tensor) (Tensor, error)
}
*/

type DescFuncOptHandler[DT any] interface {
	HandleFuncOptsDesc(a Basic[DT], expShape Shape, opts ...FuncOpt) (retVal DescWithStorage, fo Option, err error)
}

type SpecializedFuncOptHandler[DT any, T Tensor[DT, T]] interface {
	HandleFuncOptsSpecialized(a T, expShape Shape, opts ...FuncOpt) (retVal T, fo Option, err error)
}

type FuncOptHandler[DT any] interface {
	HandleFuncOpts(a Basic[DT], expShape Shape, opts ...FuncOpt) (retVal Basic[DT], fo Option, err error)
}

type Adder[DT any, T Basic[DT]] interface {
	Add(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	AddScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
	AddBroadcastable(ctx context.Context, a, b, retVal T, expShapeA, expShapeB shapes.Shape, toIncr bool) (err error)
}

type BasicArither[DT any, T Tensor[DT, T]] interface {
	Adder[DT, T]

	Sub(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	SubScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)

	Mul(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	MulScalar(ctx context.Context, a T, b DT, retVal T, scalarOnLeft, toIncr bool) (err error)

	Div(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	DivScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
}

type Arither[DT any, T Tensor[DT, T]] interface {
	BasicArither[DT, T]

	Mod(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	ModScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)

	Pow(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	PowScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
}

type Comparer[DT any, T Tensor[DT, T]] interface {
	ElEq(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	ElEqScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)

	Ne(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error)
	NeScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
}

type Ord[DT any, T Tensor[DT, T]] interface {
	Comparer[DT, T]
	Lt(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	LtScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)

	Lte(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	LteScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
}

type FullOrd[DT any, T Tensor[DT, T]] interface {
	Ord[DT, T]

	Gt(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	GtScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)

	Gte(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	GteScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
}

type MinMaxer[DT any, T Tensor[DT, T]] interface {
	MinBetween(ctx context.Context, a, b, retVal T) (err error)
	MinBetweenScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft bool) (err error)

	MaxBetween(ctx context.Context, a, b, retVal T) (err error)
	MaxBetweenScalar(ctx context.Context, t T, s DT, retval T, scalarOnLeft bool) (err error)
}

type Argmethoder[DT any, T Tensor[DT, T]] interface {
	Argmax(ctx context.Context, a T, axis int) (Basic[int], error)
	Argmin(ctx context.Context, a T, axis int) (Basic[int], error)
}

type Reducer[DT any, T Tensor[DT, T]] interface {
	Reduce(ctx context.Context, fn any, a T, axis int, defaultValue DT, retVal T) (err error)
	ReduceAlong(ctx context.Context, fns any, defaultValue DT, a T, retVal T, along ...int) (err error)

	PrepReduce(a T, opts ...FuncOpt) (ctx context.Context, axes []int, retVal T, err error)
}

type Scanner[DT any, T Tensor[DT, T]] interface {
	Scan(ctx context.Context, fn any, a T, axis int, retVal T) (err error)
}

type DotIterer[DT any, T Tensor[DT, T]] interface {
	DotIter(ctx context.Context, reduceWithFn, elwiseFn func(DT, DT) DT, a, b, retVal T) (err error)
}

type Mapper[DT any, T Tensor[DT, T]] interface {
	Map(ctx context.Context, fn any, a T, retVal T) (err error)
}

type SVDer[DT any, T Basic[DT]] interface {
	SVD(ctx context.Context, a T, uv, full bool) (s, u, v T, err error)
}

type Tracer[DT any, T Basic[DT]] interface {
	Trace(ctx context.Context, t T) (DT, error)
}

type Scatterer[DT any, T Tensor[DT, T]] interface {
	Scatter(ctx context.Context, a T, indices Basic[int], retVal T) (err error)
}

// InnerProder are any engines that support the computation of  inner products.
type InnerProder[DT any, T Basic[DT]] interface {
	Inner(ctx context.Context, a, b T) (DT, error)
}

// BLA are any engines that can support basic linear algebra
type BLA[DT any, T Basic[DT]] interface {
	InnerProder[DT, T]
	FMA(ctx context.Context, a, x, retVal T) error
	MatVecMul(ctx context.Context, a, b, retVal T, incr []DT) error
	MatMul(ctx context.Context, a, b, retVal T, incr []DT) error
	Outer(ctx context.Context, a, b, retVal T, incr []DT) error
}

type Repeater[DT any, T Tensor[DT, T]] interface {
	Repeat(ctx context.Context, a, retVal T, axis, size int, repeats []int) error
	PrepRepeat(a T, axis int, repeats []int, opts ...FuncOpt) (ctx context.Context, retVal T, newAxis, size int, newRepeats []int, err error)
}

type FancyIndexer[DT any, T Tensor[DT, T]] interface {
	SelectByIndices(ctx context.Context, a T, indices Basic[int], axis int, retVal T) (err error)
	SelectByIndicesB(ctx context.Context, input, outGrade T, indices Basic[int], axis int, retVal T) (err error)
}

type Normer[DT any, T Tensor[DT, T]] interface {
	Norm(ctx context.Context, t T, ord NormOrder, axes []int) Basic[DT]
	Norm2(ctx context.Context, t T) DT
}

type Concater[DT any, T Tensor[DT, T]] interface {
	Concat(ctx context.Context, a T, axis int, others ...T) (T, error)
}
