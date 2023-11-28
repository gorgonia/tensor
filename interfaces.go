package tensor

import (
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
)

// Memsetter is anything that can set a particular memory
type Memsetter interface {
	Memset(v any) error
}

// NonStandardEngine is any kind of engine that isn't based off the standard engines provided in this package.
type NonStandardEngine interface {
	MemoryFlag() MemoryFlag
}

/* Tensor Definitions. Oh boy this is gonna get wild */

type Engineer interface {
	Engine() Engine
}

// Desc is the minimal interface a tensor has to implement. It describes a tensor.
type Desc interface {
	// info
	Dtyper
	Shape() shapes.Shape
	Strides() []int
	Dims() int
	Size() int

	Info() *AP
}

// DescWithstorage is a minimal interface of a tensor that actually contains data.
type DescWithStorage interface {
	Desc
	DataSizer

	// Flags returns the memory flags of the underlying data array.
	Flags() MemoryFlag
	// DataOrder returns the data order of the underlying data array.
	DataOrder() DataOrder

	// Some basic operations that does not need knowledge of datatype

	// A basic tensor should be able to reshape itself
	Reshape(shape ...int) error

	// A basic tensor should be able to unsqueeze itself
	Unsqueeze(axis int) error

	// A Basic tensor should be able to zero itself out
	Zeroer

	// Data access related methods

	RequiresIterator() bool
	Iterator() Iterator
	IsMaterializable() bool

	// Memory and operation related methods

	Memory
	Engineer
	IsNativelyAccessible() bool // Can Go access the memory?
	IsManuallyManaged() bool    // Must Go manage the memory

	// Construction related

	AlikeAsTyper
	DescAliker

	// Restore restores any overallocated tensors back to the correct data legnth
	Restore()

	// SetDataOrder sets the data order of the given tensor.
	SetDataOrder(o DataOrder)
}

type Basic[DT any] interface {
	DescWithStorage

	RawAccessor[DT]
	ValueSetter[DT]

	// A Basic tensor should be able to clone itself
	BasicCloner[DT]

	// A Basic tensor should be able to create something like itself
	BasicAliker[DT]
}

type Slicer[T any] interface {
	Slice(rs ...SliceRange) (T, error)
}

// Operable represents any tensor-like structure that can perform tensor-like operations without having to know about the datatype contained within.
type Operable[T any] interface {
	DescWithStorage

	// Aliker makes sure that a Tensor can create one that is like it
	Aliker[T]

	// Eq checks that a tensor is equal to another.
	Eq[T]

	// Cloner creates a clone of the given tensor.
	Cloner[T]

	// Slice slices a tensor and returns a view
	Slice(rs ...SliceRange) (T, error)

	// T performs a thunked transposition.
	T(axes ...int) (T, error)

	// Materialize turns a view into T.
	Materialize() (T, error)

	// Repeat repeats the *Dense tensor along the given axis the given number of times.
	Repeat(axis int, repeats ...int) (T, error)
}

type Tensor[DT any, T Basic[DT]] interface {
	Basic[DT] // Must be the same as T. There's currently no way to validate this yet

	// Tensors have a tensor-like structure
	Operable[T]

	// Apply applies a scalar function
	Apply(fn any, opts ...FuncOpt) (T, error)

	// Reduce reduces the dimensions of a tensor with a given function fn.
	// You may specify the axes to reduce along with `Along`. If no axes are specified
	// then the default reduction axis is along all axes.
	Reduce(fn any, defaultValue DT, opts ...FuncOpt) (T, error)

	// Scan
	Scan(fn func(a, b DT) DT, axis int, opts ...FuncOpt) (T, error)

	// Dot
	Dot(reductionFn, elwiseFn func(DT, DT) DT, other T, opts ...FuncOpt) (T, error)
}

type RawAccessor[DT any] interface {
	Data() []DT
}

type ValueGetter[DT any] interface {
	At(...int) (DT, error)
}

type ValueSetter[DT any] interface {
	SetAt(v DT, coord ...int) error

	Memset(v DT) error
	Zeroer
}

type DataSizer interface {
	// DataSize returns the size of the underlying data. If it's overallocated it will return the size of the whole overallocated array.
	DataSize() int
}

/*
All tensors must implement these
*/

type Dtyper interface {
	Dtype() dtype.Dtype
}

type Eq[T any] interface {
	Eq(other T) bool
}

type Aliker[T any] interface {
	Alike(opts ...ConsOpt) T
}

type AlikeAsTyper interface {
	AlikeAsType(dt dtype.Dtype, opts ...ConsOpt) DescWithStorage
}

// BasicAliker is like Aliker, but the return value is Basic[DT].
//
// While this gives more flexibility, care must be taken when implementing
// tensors that embeds another tensor.
type BasicAliker[DT any] interface {
	AlikeAsBasic(opts ...ConsOpt) Basic[DT]
}

// DescAliker is like Aliker, but the return value is of DescWithStorage.
type DescAliker interface {
	AlikeAsDescWithStorage(opts ...ConsOpt) DescWithStorage
}

type Cloner[T any] interface {
	Clone() T
}

// BasicCloner is like Cloner, but the return value is a Basic[DT].
//
// Important to note that Go's type system will not check for correctness of this
// automatically, when embedding any tensor that embeds another tensor. See the wrapped_test example in package dense.
type BasicCloner[DT any] interface {
	CloneAsBasic() Basic[DT]
}

type ShallowCloner[T any] interface {
	ShallowClone() T
}

type Applyer[DT any, T Basic[DT]] interface {
	Apply(fn func(DT) (DT, error), opts ...FuncOpt) T
}

type Zeroer interface {
	Zero()
}

// Value is a Tensor-like type that only supports reading but not writing data.
//
// This allows for scalar values to be used
type Value[DT any] interface {
	Desc
	RawAccessor[DT]
	Engine() Engine
}

/* Variant Basic[DT] types */

type DenseTensor[DT any] interface {
	Basic[DT]
	IsDenseTensor()
}

type SparseTensor[DT any] interface {
	Basic[DT]
	IsSparseTensor()
}

type Scalar[DT any] interface {
	IsScalar()
	ScalarValue() DT
}
