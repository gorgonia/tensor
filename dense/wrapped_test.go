package dense

import (
	"gorgonia.org/tensor"
	"gorgonia.org/dtype"
)

var _ tensor.Tensor[float64, *Wrapped[float64]] = &Wrapped[float64]{}

var _ DenseTensor[float64, *Wrapped[float64]] = &Wrapped[float64]{}

type Wrapped[DT any] struct {
	*Dense[DT]
}

func (t *Wrapped[DT]) Apply(fn func(DT) (DT, error), opts ...FuncOpt) (*Wrapped[DT], error) {
	panic("NYI")
}

func (t *Wrapped[DT]) Clone() *Wrapped[DT] { panic("NYI") }

// CloneAsBasic: IMPORTANT NOTE: you MUST implement this correctly if you are creating a wrapped tensor.
func (t *Wrapped[DT]) CloneAsBasic() tensor.Basic[DT] { panic("NYI") }

func (t *Wrapped[DT]) Eq(other *Wrapped[DT]) bool { panic("NYI") }

func (t *Wrapped[DT]) Reduce(fn any, defaultValue DT, opts ...FuncOpt) (*Wrapped[DT], error) {
	panic("NYI")
}

func (t *Wrapped[DT]) Scan(fn func(a, b DT) DT, axis int, opts ...FuncOpt) (*Wrapped[DT], error) {
	panic("NYI")
}

func (t *Wrapped[DT]) Slice(rs ...SliceRange) (*Wrapped[DT], error) { panic("NYI") }

func (t *Wrapped[DT]) T(axes ...int) (*Wrapped[DT], error) { panic("NYI") }

func (t *Wrapped[DT]) Transpose(axes ...int) (*Wrapped[DT], error) { panic("NYI") }

func (t *Wrapped[DT]) Materialize() (*Wrapped[DT], error) { panic("NYI") }

func (t *Wrapped[DT]) Alike(opts ...ConsOpt) *Wrapped[DT] {
	return &Wrapped[DT]{t.Dense.Alike(opts...)}
}

func (t *Wrapped[DT]) AlikeAsType(dt dtype.Datatype, opts ...ConsOpt) DescWithStorage {
	panic("NYI")
}

/* Construction related methods */

func (t *Wrapped[DT]) FromDense(d *Dense[DT]) *Wrapped[DT] { return &Wrapped[DT]{d} }

func (t *Wrapped[DT]) GetDense() *Dense[DT] { return t.Dense }