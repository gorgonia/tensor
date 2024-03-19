package tensor

import (
	"context"
)

/* Data Agnostic Execution Engine Methods */

// Transposer is any engine that can perform an unsafe transpose of a tensor.
type Transposer[DT any] interface {
	Transpose(ctx context.Context, t Basic[DT], expStrides []int) error
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

type Adder[DT any] interface {
	Add(ctx context.Context, a, b, retVal Basic[DT], toIncr bool) (err error)
	AddScalar(ctx context.Context, t Basic[DT], s DT, retVal Basic[DT], scalarOnLeft, toIncr bool) (err error)
	AddBroadcastable(ctx context.Context, a, b, retVal Basic[DT], expAPA, expAPB *AP, toIncr bool) (err error)
}

type BasicArither[DT any] interface {
	Adder[DT]

	Sub(ctx context.Context, a, b, retVal Basic[DT], toIncr bool) (err error)
	SubScalar(ctx context.Context, t Basic[DT], s DT, retVal Basic[DT], scalarOnLeft, toIncr bool) (err error)
	SubBroadcastable(ctx context.Context, a, b, retVal Basic[DT], expAPA, expAPB *AP, toIncr bool) (err error)

	Mul(ctx context.Context, a, b, retVal Basic[DT], toIncr bool) (err error)
	MulScalar(ctx context.Context, a Basic[DT], b DT, retVal Basic[DT], scalarOnLeft, toIncr bool) (err error)
	MulBroadcastable(ctx context.Context, a, b, retVal Basic[DT], expAPA, expAPB *AP, toIncr bool) (err error)

	Div(ctx context.Context, a, b, retVal Basic[DT], toIncr bool) (err error)
	DivScalar(ctx context.Context, t Basic[DT], s DT, retVal Basic[DT], scalarOnLeft, toIncr bool) (err error)
	DivBroadcastable(ctx context.Context, a, b, retVal Basic[DT], expAPA, expAPB *AP, toIncr bool) (err error)
}

type Arither[DT any] interface {
	BasicArither[DT]

	Mod(ctx context.Context, a, b, retVal Basic[DT], toIncr bool) (err error)
	ModScalar(ctx context.Context, t Basic[DT], s DT, retVal Basic[DT], scalarOnLeft, toIncr bool) (err error)
	ModBroadcastable(ctx context.Context, a, b, retVal Basic[DT], expAPA, expAPB *AP, toIncr bool) (err error)

	Pow(ctx context.Context, a, b, retVal Basic[DT], toIncr bool) (err error)
	PowScalar(ctx context.Context, t Basic[DT], s DT, retVal Basic[DT], scalarOnLeft, toIncr bool) (err error)
	PowBroadcastable(ctx context.Context, a, b, retVal Basic[DT], expAPA, expAPB *AP, toIncr bool) (err error)
}

type Comparer[DT any] interface {
	ElEq(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool) (err error)
	ElEqScalar(ctx context.Context, a Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	ElEqBroadcastable(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)

	ElNe(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, asSameDT bool) (err error)
	ElNeScalar(ctx context.Context, a Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	ElNeBroadcastable(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)
}

type Ord[DT any] interface {
	Comparer[DT]
	Lt(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool) (err error)
	LtScalar(ctx context.Context, a Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	LtBroadcastable(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)

	Lte(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool) (err error)
	LteScalar(ctx context.Context, a Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	LteBroadcastable(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)
}

type FullOrd[DT any] interface {
	Ord[DT]

	Gt(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool) (err error)
	GtScalar(ctx context.Context, a Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	GtBroadcastable(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)

	Gte(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool) (err error)
	GteScalar(ctx context.Context, a Basic[DT], b DT, retVal DescWithStorage, scalarOnLeft bool, returnSameDataType bool) (err error)
	GteBroadcastable(ctx context.Context, a, b Basic[DT], retVal DescWithStorage, returnSameDataType bool, expAPA, expAPB *AP) (err error)
}

type MinMaxer[DT any] interface {
	MinBetween(ctx context.Context, a, b, retVal Basic[DT]) (err error)
	MinBetweenScalar(ctx context.Context, t Basic[DT], s DT, retVal Basic[DT], scalarOnLeft bool) (err error)

	MaxBetween(ctx context.Context, a, b, retVal Basic[DT]) (err error)
	MaxBetweenScalar(ctx context.Context, t Basic[DT], s DT, retval Basic[DT], scalarOnLeft bool) (err error)
}

type Argmethoder[DT any] interface {
	Argmax(ctx context.Context, a Basic[DT], axis int) (Basic[int], error)
	Argmin(ctx context.Context, a Basic[DT], axis int) (Basic[int], error)
}

type Reducer[DT any] interface {
	Reduce(ctx context.Context, fn any, a Basic[DT], axis int, defaultValue DT, retVal Basic[DT]) (err error)
	ReduceAlong(ctx context.Context, fns any, defaultValue DT, a Basic[DT], retVal Basic[DT], along ...int) (err error)

	PrepReduce(a Basic[DT], opts ...FuncOpt) (ctx context.Context, axes []int, retVal Basic[DT], err error)
}

type Scanner[DT any] interface {
	Scan(ctx context.Context, fn any, a Basic[DT], axis int, retVal Basic[DT]) (err error)
}

type DotIterer[DT any] interface {
	DotIter(ctx context.Context, reduceWithFn, elwiseFn func(DT, DT) DT, a, b, retVal Basic[DT]) (err error)
}

type Mapper[DT any] interface {
	Map(ctx context.Context, fn any, a Basic[DT], retVal Basic[DT]) (err error)
}

type SVDer[DT any, T Basic[DT]] interface {
	SVD(ctx context.Context, a T, uv, full bool) (s, u, v T, err error)
}

type Tracer[DT any] interface {
	Trace(ctx context.Context, t Basic[DT]) (DT, error)
}

type Scatterer[DT any] interface {
	Scatter(ctx context.Context, a Basic[DT], indices Basic[int], retVal Basic[DT]) (err error)
}

// InnerProder are any engines Basic[DT]hat support Basic[DT]he computation of  inner products.
type InnerProder[DT any] interface {
	Inner(ctx context.Context, a, b Basic[DT]) (DT, error)
}

// BLA are any engines Basic[DT]hat can support basic linear algebra
type BLA[DT any] interface {
	InnerProder[DT]
	FMA(ctx context.Context, a, x, retVal Basic[DT]) error
	MatVecMul(ctx context.Context, a, b, retVal Basic[DT], incr []DT) error
	MatMul(ctx context.Context, a, b, retVal Basic[DT], incr []DT) error
	Outer(ctx context.Context, a, b, retVal Basic[DT], incr []DT) error
}

type Repeater[DT any, T Tensor[DT, T]] interface {
	Repeat(ctx context.Context, a, retVal T, axis, size int, repeats []int) error
	PrepRepeat(a T, axis int, repeats []int, opts ...FuncOpt) (ctx context.Context, retVal T, newAxis, size int, newRepeats []int, err error)
}

type FancyIndexer[DT any, T Tensor[DT, T]] interface {
	SelectByIndices(ctx context.Context, a T, indices Basic[int], axis int, retVal T) (err error)
	SelectByIndicesB(ctx context.Context, input, outGrad T, indices Basic[int], axis int, retVal T) (err error)
}

type Normer[DT any] interface {
	Norm(ctx context.Context, t Basic[DT], ord NormOrder, axes []int) Basic[DT]
	Norm2(ctx context.Context, t Basic[DT]) DT
}

type Concater[DT any, T Basic[DT]] interface {
	Concat(ctx context.Context, a T, axis int, others ...Basic[DT]) (T, error)
}

/*
Unary Operations

Note: Basic[DT]ypically Basic[DT]hese unary operations would have been supported by means of .Apply()
but .Apply() cannot work for CUDA related stuff, so Basic[DT]hese unary operation interfaces
are added for Basic[DT]he usual deep learning related engines.
*/

type Abser[DT Num] interface {
	Abs(ctx context.Context, a, retVal Basic[DT]) error
}

type Signer[DT Num] interface {
	Sign(ctx context.Context, a, retVal Basic[DT]) error
}

type Ceiler[DT Floats] interface {
	Ceil(ctx context.Context, a, retVal Basic[DT]) error
}

type Floorer[DT Floats] interface {
	Floor(ctx context.Context, a, retVal Basic[DT]) error
}

type Neger[DT Num] interface {
	Neg(ctx context.Context, a, retVal Basic[DT]) error
}

type ExpLoger[DT Floats] interface {
	Exp(ctx context.Context, a, retVal Basic[DT]) error
	Log(ctx context.Context, a, retVal Basic[DT]) error
	Log2(ctx context.Context, a, retVal Basic[DT]) error
	Log10(ctx context.Context, a, retVal Basic[DT]) error
	Log1p(ctx context.Context, a, retVal Basic[DT]) error
	Expm1(ctx context.Context, a, retVal Basic[DT]) error
}

type Inver[DT Num] interface {
	Inv(ctx context.Context, a, retVal Basic[DT]) error
}

type InverSqrter[DT Floats] interface {
	InvSqrt(ctx context.Context, a, retVal Basic[DT]) error
}

type Squarer[DT Num] interface {
	Square(ctx context.Context, a, retVal Basic[DT]) error
}

type Sqrter[DT Floats] interface {
	Sqrt(ctx context.Context, a, retVal Basic[DT]) error
}

type Cuber[DT Num] interface {
	Cube(ctx context.Context, a, retVal Basic[DT]) error
}

type Tanher[DT Floats] interface {
	Tanh(ctx context.Context, a, retVal Basic[DT]) error
}
