package sparse

import "gorgonia.org/tensor"

func (t *CS[DT]) Slice(slices ...tensor.SliceRange) (*CS[DT], error) {
	panic("NYI")
}

func (t *CS[DT]) T(axes ...int) (*CS[DT], error) {
	panic("NYI")
}
