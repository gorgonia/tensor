package dense

import (
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

// metaops.go contains methods that perform operation on metadata of the *Dense tensor

func (t *Dense[T]) Reshape(dims ...int) error {
	if len(t.data) < shapes.Shape(dims).TotalSize() {
		return errors.Errorf("Cannot reshape a tensor with %d elements (as %v) into %v", len(t.data), t.Shape(), dims)
	}
	t.AP.SetShape(dims...)
	t.fix()
	return nil
}

// Unsqueeze adds a new axis at the specified position
func (t *Dense[T]) Unsqueeze(axis int) error {
	if axis < 0 || axis > t.Dims() {
		return errors.Errorf("Cannot unsqueeze %v at axis %d", t.Shape(), axis)
	}
	shp := t.Shape()
	shp = append(shp[:axis], 1)
	copy(shp[axis+1:], shp[axis:])
	shp[axis] = 1

	strides := t.Strides()
	strides = append(strides[:axis], 1)
	copy(strides[axis+1:], strides[axis:])
	t.AP.SetShape(shp...)
	t.AP.SetStrides(strides)
	return nil
}

// T performs a thunked transpose. It doesn't actually do anything, except store extra information about the post-transposed shapes and strides
// Usually this is more than enough, as BLAS will handle the rest of the transpose
func (t *Dense[DT]) T(axes ...int) (view *Dense[DT], err error) {
	view = t.ShallowClone()

	var transform AP
	if transform, axes, err = view.AP.T(axes...); err != nil {
		return view, err
	}
	view.AP = transform
	view.f = view.f.ViewFlag()
	view.transposedWith = axes
	return view, nil

	/*
		// is there any old transposes that need to be done first?
		// this is important, because any old transposes for dim >=3 are merely permutations of the strides
		if !t.old.IsZero() {
			if t.IsVector() {
				// the transform that was calculated was a waste of time - return it to the pool then untranspose
				t.UT()
				return
			}

			// check if the current axes are just a reverse of the previous transpose's
			isReversed := true
			for i, s := range t.oshape() {
				if transform.Shape()[i] != s {
					isReversed = false
					break
				}
			}

			// if it is reversed, well, we just restore the backed up one
			if isReversed {
				t.UT()
				return
			}

			// cool beans. No funny reversals. We'd have to actually do transpose then
			t.Transpose()
		}

		// swap out the old and the new
		t.old = t.AP
		t.transposeWith = axes
		t.AP = transform
		return nil
	*/
}
func (t *Dense[DT]) Slice(slices ...SliceRange) (retVal *Dense[DT], err error) {
	var newAP AP
	var ndStart, ndEnd int
	if newAP, ndStart, ndEnd, err = t.AP.S(len(t.data), slices...); err != nil {
		return
	}
	if ndStart < 0 || ndEnd < ndStart || ndEnd > cap(t.data) {
		return nil, errors.Errorf("Cannot slice %T. Index %d:%d is out of bounds", t, ndStart, ndEnd)
	}

	v := &Dense[DT]{}
	v.copyMetadata(newAP, t.e, t.f.ViewFlag(), t.t)
	// TODO: consider when the data is no longer contiguous

	v.data = t.data[ndStart:ndEnd]
	v.bytes = t.bytes[ndStart*int(t.t.Size()) : ndEnd*int(t.t.Size())]

	// TODO: slicing non accessible data

	return v, nil
}

// RollAxis rolls the axis backwards until it lies in the given position.
//
// This method was adapted from Numpy's Rollaxis. The licence for Numpy is a BSD-like licence and can be found here:
// https://github.com/numpy/numpy/blob/master/LICENSE.txt
//
// As a result of being adapted from Numpy, the quirks are also adapted. A good guide reducing the confusion around rollaxis can be found here:
// http://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing (see answer by hpaulj)
func (t *Dense[DT]) RollAxis(axis, start int, safe bool) (retVal *Dense[DT], err error) {
	dims := t.Dims()

	if !(axis >= 0 && axis < dims) {
		err = errors.Errorf(errors.InvalidAxis, axis, dims)
		return
	}

	if !(start >= 0 && start <= dims) {
		err = errors.Wrap(errors.Errorf(errors.InvalidAxis, axis, dims), "Start axis is wrong")
		return
	}

	if axis < start {
		start--
	}

	if axis == start {
		retVal = t
		return
	}

	axes := make([]int, dims)

	for i := 0; i < dims; i++ {
		axes[i] = i
	}
	copy(axes[axis:], axes[axis+1:])
	copy(axes[start+1:], axes[start:])
	axes[start] = axis

	if safe {
		return t.T(axes...)
	}
	err = t.UnsafeT(axes...)
	retVal = t
	return
}

// Narrow narrows the given dimension of a tensor.
func (t *Dense[DT]) Narrow(dim, start, length int) (*Dense[DT], error) {
	dim = resolveAxis(dim, t.Dims())

	slices := make([]SliceRange, internal.Min(dim+1, t.Dims()))
	slices[dim] = SR(start, start+length, 1)

	return t.Slice(slices...)
}

func (t *Dense[DT]) UnsafeT(axes ...int) (err error) {
	var transform AP
	if transform, axes, err = t.AP.T(axes...); err != nil {
		return err
	}
	t.AP = transform
	return nil
}
