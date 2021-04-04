package tensor

// a View is a *Tensor with customized strides. The reason for not splitting them up into different types is complicated
// this file contains all the methods that deals with Views

type DenseView struct {
	*Dense
}

// RequiresIterator returns true if an iterator is required to read the data in the correct fashion.
func (t DenseView) RequiresIterator() bool {
	if t.len() == 1 {
		return false
	}
	// non continuous slice, transpose, or masked. If it's a slice and contiguous, then iterator is not required
	if !t.o.IsContiguous() || !t.old.IsZero() || t.IsMasked() {
		return true
	}
	return false
}

// IsView indicates if the Tensor is a view of another (typically from slicing)
func (t DenseView) IsView() bool {
	return t.viewOf != 0
}

// IsMaterializeable indicates if the Tensor is materializable - if it has either gone through some transforms or slicing
func (t DenseView) IsMaterializable() bool {
	return t.viewOf != 0 || !t.old.IsZero()
}

// Materialize takes a view, copies its data and puts it in a new *Tensor.
func (t DenseView) Materialize() Tensor {
	if !t.IsMaterializable() {
		return t
	}

	retVal := recycledDense(t.t, t.shape.Clone(), WithEngine(t.e))
	copyDenseIter(retVal, t, nil, nil)
	retVal.e = t.e
	retVal.oe = t.oe
	return retVal
}
