package scalar

import (
	"github.com/pkg/errors"
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
)

type Scalar[T any] struct {
	V T
}

func S[T any](a T) Scalar[T] {
	return Scalar[T]{V: a}
}

func (s Scalar[T]) Info() *AP          { return nil }
func (s Scalar[T]) Dtype() dtype.Dtype { return dtype.Datatype[T]{} }

func (s Scalar[T]) Shape() shapes.Shape { return shapes.ScalarShape() }

func (s Scalar[T]) Strides() []int { return nil }

func (s Scalar[T]) Dims() int { return 0 }

func (s Scalar[T]) Size() int { return 0 }

func (s Scalar[T]) DataSize() int { return 1 }

func (s Scalar[T]) Data() []T { return []T{s.V} }

func (s Scalar[T]) ScalarValue() T { return s.V }

func (s Scalar[T]) At(coords ...int) (T, error) {
	return s.V, errors.Errorf("You can't do .At() on a Scalar")
}

func (s Scalar[T]) Engine() Engine { return nil }

/*
func (s Scalar[T]) RequiresIterator() bool { return false }
func (s Scalar[T]) Iterator() Iterator     { return nil }
func (s Scalar[T]) DataOrder() DataOrder   { panic("Scalars have no DataOrder") }

func (s Scalar[T]) Engine() Engine             { return nil }
func (s Scalar[T]) IsNativelyAccessible() bool { return true }
func (s Scalar[T]) IsManuallyManaged() bool   { return false }

func (s Scalar[T]) MemSize() uintptr {panic("NYI")}
func (s Scalar[T]) Uintptr() uintptr {panic("NYI")}



// BAD

func (s Scalar[T]) SetAt(v T, coord ...int) error {return errors.New("Cannot set scalar values" )}
func (s Scalar[T]) Memset(v T) error {return errors.New("Cannot memset scalar")}
func (s Scalar[T]) Zero() {}
*/
