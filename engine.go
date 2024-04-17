package tensor

import (
	"context"
)

/* Data Agnostic Execution Engine Methods */

// Transposer is any engine that can perform an unsafe transpose of a tensor.
type Transposer[DT any, T Tensor[DT, T]] interface {
	Transpose(ctx context.Context, t T, expStrides []int) error
}

type Copier[DT any, T Tensor[DT, T]] interface {
	Copy(ctx context.Context, dst, src T) error
}

/*
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
	AddBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *AP, toIncr bool) (err error)
}

type BasicArither[DT any, T Basic[DT]] interface {
	Adder[DT, T]

	Sub(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	SubScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
	SubBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *AP, toIncr bool) (err error)

	Mul(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	MulScalar(ctx context.Context, a T, b DT, retVal T, scalarOnLeft, toIncr bool) (err error)
	MulBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *AP, toIncr bool) (err error)

	Div(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	DivScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
	DivBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *AP, toIncr bool) (err error)
}

type Arither[DT any, T Basic[DT]] interface {
	BasicArither[DT, T]

	Mod(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	ModScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
	ModBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *AP, toIncr bool) (err error)

	Pow(ctx context.Context, a, b, retVal T, toIncr bool) (err error)
	PowScalar(ctx context.Context, t T, s DT, retVal T, scalarOnLeft, toIncr bool) (err error)
	PowBroadcastable(ctx context.Context, a, b, retVal T, expAPA, expAPB *AP, toIncr bool) (err error)
}

type Comparer[DT any, T Basic[DT]] interface {
	ElEq(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	ElEqScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	ElEqBroadcastable(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)

	ElNe(ctx context.Context, a, b T, retVal DescWithStorage, asSameDT bool) (err error)
	ElNeScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	ElNeBroadcastable(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)
}

type Ord[DT any, T Basic[DT]] interface {
	Comparer[DT, T]
	Lt(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	LtScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	LtBroadcastable(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)

	Lte(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	LteScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	LteBroadcastable(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)
}

type FullOrd[DT any, T Basic[DT]] interface {
	Ord[DT, T]

	Gt(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	GtScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	GtBroadcastable(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)

	Gte(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool) (err error)
	GteScalar(ctx context.Context, a T, b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	GteBroadcastable(ctx context.Context, a, b T, retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)
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
	PrepReduce(a T, opts ...FuncOpt) (fo Option, axes []int, retVal T, err error)
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

type Stacker[DT any, T Tensor[DT, T]] interface {
	Stack(ctx context.Context, a T, axis int, others ...T) (T, error)
}

/*
Unary Operations

Note: typically these unary operations would have been supported by means of .Apply()
but .Apply() cannot work for CUDA related stuff, so these unary operation interfaces
are added for the usual deep learning related engines.
*/

type Abser[DT Num, T Basic[DT]] interface {
	Abs(ctx context.Context, a, retVal T) error
}

type Signer[DT Num, T Basic[DT]] interface {
	Sign(ctx context.Context, a, retVal T) error
}

type Ceiler[DT Floats, T Basic[DT]] interface {
	Ceil(ctx context.Context, a, retVal T) error
}

type Floorer[DT Floats, T Basic[DT]] interface {
	Floor(ctx context.Context, a, retVal T) error
}

type Neger[DT Num, T Basic[DT]] interface {
	Neg(ctx context.Context, a, retVal T) error
}

type ExpLoger[DT Floats, T Basic[DT]] interface {
	Exp(ctx context.Context, a, retVal T) error
	Log(ctx context.Context, a, retVal T) error
	Log2(ctx context.Context, a, retVal T) error
	Log10(ctx context.Context, a, retVal T) error
	Log1p(ctx context.Context, a, retVal T) error
	Expm1(ctx context.Context, a, retVal T) error
}

type Inver[DT Num, T Basic[DT]] interface {
	Inv(ctx context.Context, a, retVal T) error
}

type InverSqrter[DT Floats, T Basic[DT]] interface {
	InvSqrt(ctx context.Context, a, retVal T) error
}

type Squarer[DT Num, T Basic[DT]] interface {
	Square(ctx context.Context, a, retVal T) error
}

type Sqrter[DT Floats, T Basic[DT]] interface {
	Sqrt(ctx context.Context, a, retVal T) error
}

type Cuber[DT Num, T Basic[DT]] interface {
	Cube(ctx context.Context, a, retVal T) error
}

type Tanher[DT Floats, T Basic[DT]] interface {
	Tanh(ctx context.Context, a, retVal T) error
}
