package scalar

import (
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

// some constraints
var (
	_ DescWithStorage = S[int](1337)
)

type Scalar[DT any] struct {
	V DT
}

// S creates a new Scalar from the value
func S[DT any](a DT) Scalar[DT] {
	return Scalar[DT]{V: a}
}

func Z(dt dtype.Dtype) DescWithStorage { panic("XXX") }

func (s Scalar[DT]) Info() *AP          { return nil }
func (s Scalar[DT]) Dtype() dtype.Dtype { return dtype.Datatype[DT]{} }

func (s Scalar[DT]) Shape() shapes.Shape { return shapes.ScalarShape() }

func (s Scalar[DT]) Strides() []int { return nil }

func (s Scalar[DT]) Dims() int { return 0 }

func (s Scalar[DT]) Size() int { return 0 }

func (s Scalar[DT]) DataSize() int { return 1 }

func (s Scalar[DT]) Data() []DT { return []DT{s.V} }

func (s Scalar[DT]) ScalarValue() DT { return s.V }

func (s Scalar[DT]) At(coords ...int) (DT, error) {
	return s.V, errors.Errorf("You can't do .At() on a Scalar")
}

func (s Scalar[DT]) Engine() Engine { return nil }

func (s Scalar[DT]) AlikeAsDescWithStorage(opts ...ConsOpt) DescWithStorage { return s }

func (s Scalar[DT]) AlikeAsType(dt dtype.Dtype, opts ...ConsOpt) DescWithStorage {
	switch dt {
	default:
		panic("NYI: put a pull request")
	}
}

func (s Scalar[DT]) DataOrder() DataOrder       { return ColMajor }
func (s Scalar[DT]) Flags() MemoryFlag          { return 0 }
func (s Scalar[DT]) IsNativelyAccessible() bool { return true }
func (s Scalar[DT]) IsManuallyManaged() bool    { return false }
func (s Scalar[DT]) IsMaterializable() bool     { return false }
func (s Scalar[DT]) RequiresIterator() bool     { return false }
func (s Scalar[DT]) Iterator() tensor.Iterator  { return nil }
func (s Scalar[DT]) MemSize() uintptr           { panic("NYI") }
func (s Scalar[DT]) Uintptr() uintptr           { panic("NYI") }

func (s Scalar[DT]) Reshape(...int) error       { return errors.NoOp{} }
func (s Scalar[DT]) Restore()                   {}
func (s Scalar[DT]) SetDataOrder(ord DataOrder) {}

func (s Scalar[DT]) Unsqueeze(_ int) error { return errors.NoOp{} }

func (s Scalar[DT]) Zero() {}

/*
func (s Scalar[DT]) RequiresIterator() bool { return false }
func (s Scalar[DT]) Iterator() Iterator     { return nil }





func (s Scalar[DT]) Uintptr() uintptr {panic("NYI")}



// BAD

func (s Scalar[DT]) SetAt(v T, coord ...int) error {return errors.New("Cannot set scalar values" )}
func (s Scalar[DT]) Memset(v T) error {return errors.New("Cannot memset scalar")}
func (s Scalar[DT]) Zero() {}
*/
