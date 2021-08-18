package tensor

import (
	"reflect"

	"gorgonia.org/dtype"
	"gorgonia.org/tensor/internal/storage"
)

// Dtyper is any type that has a Dtype
type Dtyper interface {
	Dtype() dtype.Dtype
}

// Eq is any type where you can perform an equality test
type Eq interface {
	Eq(interface{}) bool
}

// Cloner is any type that can clone itself
type Cloner interface {
	Clone() interface{}
}

// Dataer is any type that returns the data in its original form (typically a Go slice of something)
type Dataer interface {
	Data() interface{}
}

// Boolable is any type has a zero and one value, and is able to set itself to either
type Boolable interface {
	Zeroer
	Oner
}

// A Zeroer is any type that can set itself to the zeroth value. It's used to implement the arrays
type Zeroer interface {
	Zero()
}

// A Oner is any type that can set itself to the equivalent of one. It's used to implement the arrays
type Oner interface {
	One()
}

// A MemSetter is any type that can set itself to a value.
type MemSetter interface {
	Memset(interface{}) error
}

// A Densor is any type that can return a *Dense
type Densor interface {
	Dense() *Dense
}

// ScalarRep is any Tensor that can represent a scalar
type ScalarRep interface {
	IsScalar() bool
	ScalarValue() interface{}
}

// View is any Tensor that can provide a view on memory
type View interface {
	Tensor
	IsView() bool
	IsMaterializable() bool
	Materialize() Tensor
}

// Slicer is any tensor that can slice
type Slicer interface {
	Slice(...Slice) (View, error)
}

// Reslicer is any tensor that can reslice.
// To reslice is to reuse the container (*Dense, *CS) etc, but with new `Slice`s applied to it.
//
// e.g: A is a (3,3) matrix that has been sliced at [1:3, 1:3]. Call it B. So now B's shape is (2,2).
// B.Reslice(S(0,2), S(0,2)) would reslice the original tensor (A) with the new slices.
type Reslicer interface {
	Reslice(...Slice) (View, error)
}

// DenseTensor is the interface for any Dense tensor.
type DenseTensor interface {
	Tensor
	Info() *AP

	IsMatrix() bool
	IsVector() bool
	IsRowVec() bool
	IsColVec() bool

	// headerer
	// arrayer
	unsafeMem
	setAP(*AP)
	rtype() reflect.Type
	reshape(dims ...int) error

	setDataOrder(o DataOrder)
	isTransposed() bool
	ostrides() []int
	oshape() Shape
	transposeAxes() []int
	transposeIndex(i int, transposePat, strides []int) int
	oldAP() *AP
	setOldAP(ap *AP)
	parentTensor() *Dense
	setParentTensor(*Dense)
	len() int
	cap() int

	// operations
	Inner(other Tensor, opts ...FuncOpt) (retVal interface{}, err error)
	MatMul(other Tensor, opts ...FuncOpt) (retVal *Dense, err error)
	MatVecMul(other Tensor, opts ...FuncOpt) (retVal *Dense, err error)
	TensorMul(other Tensor, axesA, axesB []int) (retVal *Dense, err error)
	stackDense(axis int, others ...DenseTensor) (DenseTensor, error)
}

type SparseTensor interface {
	Sparse
	AsCSC()
	AsCSR()
	Indices() []int
	Indptr() []int

	// headerer
}

type MaskedTensor interface {
	DenseTensor
	IsMasked() bool
	SetMask([]bool)
	Mask() []bool
}

// Kinder. Bueno.
type Kinder interface {
	Kind() reflect.Kind
}

// MakeAliker is any Tensor that can make more like itself.
type MakeAliker interface {
	MakeAike(opts ...ConsOpt) Tensor
}

type headerer interface {
	hdr() *storage.Header
}

type arrayer interface {
	arr() array
	arrPtr() *array
}

type unsafeMem interface {
	Set(i int, x interface{})
	GetF64(i int) float64
	GetF32(i int) float32
	Float64s() []float64
	Float32s() []float32
	Complex64s() []complex64
	Complex128s() []complex128
}

type float64ser interface {
	Float64s() []float64
}

type float32ser interface {
	Float32s() []float32
}
