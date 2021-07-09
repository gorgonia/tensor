// Package tensor is a package that provides efficient, generic n-dimensional arrays in Go.
// Also in this package are functions and methods that are used commonly in arithmetic, comparison and linear algebra operations.
package tensor // import "gorgonia.org/tensor"

import (
	"encoding/gob"

	"github.com/pkg/errors"
	"gorgonia.org/dtype"
)

var (
	_ Tensor = &Dense{}
	_ Tensor = &CS{}
	_ View   = &DenseView{}
)

func init() {
	gob.Register(&Dense{})
	gob.Register(&CS{})
}

// Desc is a description of a tensor. It does not actually deal with data.
type Desc interface {
	// info about the ndarray
	Shape() Shape
	Strides() []int
	Dtype() dtype.Dtype

	Dims() int
	Size() int
	DataSize() int
}

// Tensor represents a variety of n-dimensional arrays. The most commonly used tensor is the Dense tensor.
// It can be used to represent a vector, matrix, 3D matrix and n-dimensional tensors.
type Tensor interface {
	Desc

	// Data access related
	RequiresIterator() bool
	Iterator() Iterator
	DataOrder() DataOrder

	// ops
	Slicer
	At(...int) (interface{}, error)
	SetAt(v interface{}, coord ...int) error
	Reshape(...int) error
	T(axes ...int) error
	UT()
	Transpose() error // Transpose actually moves the data
	Apply(fn interface{}, opts ...FuncOpt) (Tensor, error)

	// data related interface
	Zeroer
	MemSetter
	Dataer
	Eq
	Cloner

	// engine/memory related stuff
	// all Tensors should be able to be expressed of as a slab of memory
	// Note: the size of each element can be acquired by T.Dtype().Size()
	Memory                      // Tensors all implement Memory
	Engine() Engine             // Engine can be nil
	IsNativelyAccessible() bool // Can Go access the memory
	IsManuallyManaged() bool    // Must Go manage the memory

	// formatters
	// fmt.Formatter
	// fmt.Stringer

	// all Tensors are serializable to these formats
	//WriteNpy(io.Writer) error
	//ReadNpy(io.Reader) error
	//gob.GobEncoder
	//gob.GobDecoder

	headerer
	arrayer

	// TO BE DEPRECATED
	ScalarRep
}

// New creates a new Dense Tensor. For sparse arrays use their relevant construction function
func New(opts ...ConsOpt) *Dense {
	d := borrowDense()
	for _, opt := range opts {
		opt(d)
	}
	d.fix()
	if err := d.sanity(); err != nil {
		panic(err)
	}

	return d
}

// MustGetDense gets a *Dense from a given Tensor. Panics otherwise.
func MustGetDense(T Tensor) *Dense {
	d, err := assertDense(T)
	if err != nil {
		panic(err)
	}
	return d
}

func assertDense(t Tensor) (*Dense, error) {
	if t == nil {
		return nil, errors.New("nil is not a *Dense")
	}
	switch tt := t.(type) {
	case *Dense:
		return tt, nil
	case DenseView:
		return tt.Dense, nil
	case Densor:
		return tt.Dense(), nil
	}
	return nil, errors.Errorf("%T is not *Dense", t)
}

func getDenseTensor(t Tensor) (DenseTensor, error) {
	switch tt := t.(type) {
	case DenseTensor:
		return tt, nil
	case Densor:
		return tt.Dense(), nil
	default:
		return nil, errors.Errorf("Tensor %T is not a DenseTensor", t)
	}
}

// getFloatDense extracts a *Dense from a Tensor and ensures that the .data is a Array that implements Float
func getFloatDenseTensor(t Tensor) (retVal DenseTensor, err error) {
	if t == nil {
		return
	}
	if err = dtype.TypeClassCheck(t.Dtype(), dtype.Floats); err != nil {
		err = errors.Wrapf(err, "getFloatDense only handles floats. Got %v instead", t.Dtype())
		return
	}

	if retVal, err = getDenseTensor(t); err != nil {
		err = errors.Wrapf(err, opFail, "getFloatDense")
		return
	}
	if retVal == nil {
		return
	}

	return
}

// getFloatDense extracts a *Dense from a Tensor and ensures that the .data is a Array that implements Float
func getFloatComplexDenseTensor(t Tensor) (retVal DenseTensor, err error) {
	if t == nil {
		return
	}
	if err = dtype.TypeClassCheck(t.Dtype(), dtype.FloatComplex); err != nil {
		err = errors.Wrapf(err, "getFloatDense only handles floats and complex. Got %v instead", t.Dtype())
		return
	}

	if retVal, err = getDenseTensor(t); err != nil {
		err = errors.Wrapf(err, opFail, "getFloatDense")
		return
	}
	if retVal == nil {
		return
	}

	return
}

// sliceDense returns a *Dense.
func sliceDense(t *Dense, slices ...Slice) (retVal *Dense, err error) {
	var sliced Tensor
	if sliced, err = t.Slice(slices...); err != nil {
		return nil, err
	}
	return sliced.(DenseView).Dense, nil
}
