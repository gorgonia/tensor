package paginated

import "gorgonia.org/tensor"

// Shape will return the dimensions of the tensor.
func (p *Tensor) Shape() tensor.Shape {
	return p.dims
}

// Strides will return the strides of the paginated tensor
func (p *Tensor) Strides() []int {
	strides := make([]int, p.Dims(), p.Dims())

	for i := 0; i < p.Dims(); i++ {
		switch i {
		case 0:
			strides[i] = p.rowSize()
		case 1:
			strides[i] = p.columnSize()
		default:
			strides[i] = p.dimensionSize(i)
		}
	}

	return strides
}

// Dtype will return the data type of the tensor.
func (p *Tensor) Dtype() tensor.Dtype {
	return p.dtype
}

// Dims will return the Shape of the tensor.
func (p *Tensor) Dims() int {
	return len(p.dims)
}

// Size will return the number of values
// in the tensor.
func (p *Tensor) Size() int {
	return p.dims.TotalSize()
}

// DataSize is the same as the Size method.
func (p *Tensor) DataSize() int {
	var size int

	for _, page := range p.pages {
		size += page.Datasize
	}

	return size
}
