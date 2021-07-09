package tensor

import (
	"reflect"

	"github.com/pkg/errors"
	"gorgonia.org/dtype"
	"gorgonia.org/tensor/internal/storage"
	// "log"
)

func handleFuncOpts(expShape Shape, expType dtype.Dtype, o DataOrder, strict bool, opts ...FuncOpt) (reuse DenseTensor, safe, toReuse, incr, same bool, err error) {
	fo := ParseFuncOpts(opts...)

	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	same = fo.Same()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDenseTensor(reuseT); err != nil {
			returnOpOpt(fo)
			err = errors.Wrapf(err, "Cannot reuse a Tensor that isn't a DenseTensor. Got %T instead", reuseT)
			return
		}

		if reuse != nil && !reuse.IsNativelyAccessible() {
			returnOpOpt(fo)
			err = errors.Errorf(inaccessibleData, reuse)
			return
		}

		if (strict || same) && reuse.Dtype() != expType {
			returnOpOpt(fo)
			err = errors.Errorf(typeMismatch, expType, reuse.Dtype())
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != expShape.TotalSize() && !expShape.IsScalar() {
			returnOpOpt(fo)
			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch - reuse.len() %v, expShape.TotalSize() %v", reuse.len(), expShape.TotalSize())
			return
		}
		if !reuse.Shape().Eq(expShape) {
			cloned := expShape.Clone()
			if err = reuse.Reshape(cloned...); err != nil {
				return

			}
			ReturnInts([]int(cloned))
		}

		if !incr && reuse != nil {
			reuse.setDataOrder(o)
			// err = reuse.reshape(expShape...)
		}

	}
	returnOpOpt(fo)
	return
}

func binaryCheck(a, b Tensor, tc *typeclass) (err error) {
	// check if the tensors are accessible
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}

	if !b.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, b)
	}

	at := a.Dtype()
	bt := b.Dtype()
	if tc != nil {
		if err = typeclassCheck(at, tc); err != nil {
			return errors.Wrapf(err, typeclassMismatch, "a")
		}
		if err = typeclassCheck(bt, tc); err != nil {
			return errors.Wrapf(err, typeclassMismatch, "b")
		}
	}

	if at.Kind() != bt.Kind() {
		return errors.Errorf(typeMismatch, at, bt)
	}
	if !a.Shape().Eq(b.Shape()) {
		return errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
	}
	return nil
}

func unaryCheck(a Tensor, tc *typeclass) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	at := a.Dtype()
	if tc != nil {
		if err := typeclassCheck(at, tc); err != nil {
			return errors.Wrapf(err, typeclassMismatch, "a")
		}
	}
	return nil
}

// scalarDtypeCheck checks that a scalar value has the same dtype as the dtype of a given tensor.
func scalarDtypeCheck(a Tensor, b interface{}) error {
	var dt dtype.Dtype
	switch bt := b.(type) {
	case Dtyper:
		dt = bt.Dtype()
	default:
		t := reflect.TypeOf(b)
		dt = dtype.Dtype{t}
	}

	if a.Dtype() != dt {
		return errors.Errorf("Expected scalar to have the same Dtype as the tensor (%v). Got %T instead ", a.Dtype(), b)
	}
	return nil
}

// prepDataVV prepares the data given the input and reuse tensors. It also retruns several indicators
//
// useIter indicates that the iterator methods should be used.
// swap indicates that the operands are swapped.
func prepDataVV(a, b Tensor, reuse Tensor) (dataA, dataB, dataReuse *storage.Header, ait, bit, iit Iterator, useIter, swap bool, err error) {
	// get data
	dataA = a.hdr()
	dataB = b.hdr()
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	// iter
	useIter = a.RequiresIterator() ||
		b.RequiresIterator() ||
		(reuse != nil && reuse.RequiresIterator()) ||
		!a.DataOrder().HasSameOrder(b.DataOrder()) ||
		(reuse != nil && (!a.DataOrder().HasSameOrder(reuse.DataOrder()) || !b.DataOrder().HasSameOrder(reuse.DataOrder())))
	if useIter {
		ait = a.Iterator()
		bit = b.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
	}

	// swap
	if _, ok := a.(*CS); ok {
		if _, ok := b.(DenseTensor); ok {
			swap = true
			dataA, dataB = dataB, dataA
			ait, bit = bit, ait
		}
	}

	return
}

func prepDataVS(a Tensor, b interface{}, reuse Tensor) (dataA, dataB, dataReuse *storage.Header, ait, iit Iterator, useIter bool, newAlloc bool, err error) {
	// get data
	dataA = a.hdr()
	dataB, newAlloc = scalarToHeader(b)
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	if a.IsScalar() {
		return
	}
	useIter = a.RequiresIterator() ||
		(reuse != nil && reuse.RequiresIterator()) ||
		(reuse != nil && !reuse.DataOrder().HasSameOrder(a.DataOrder()))
	if useIter {
		ait = a.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
	}
	return
}

func prepDataSV(a interface{}, b Tensor, reuse Tensor) (dataA, dataB, dataReuse *storage.Header, bit, iit Iterator, useIter bool, newAlloc bool, err error) {
	// get data
	dataA, newAlloc = scalarToHeader(a)
	dataB = b.hdr()
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	// get iterator
	if b.IsScalar() {
		return
	}
	useIter = b.RequiresIterator() ||
		(reuse != nil && reuse.RequiresIterator()) ||
		(reuse != nil && !reuse.DataOrder().HasSameOrder(b.DataOrder()))

	if useIter {
		bit = b.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
	}
	return
}

func prepDataUnary(a Tensor, reuse Tensor) (dataA, dataReuse *storage.Header, ait, rit Iterator, useIter bool, err error) {
	// get data
	dataA = a.hdr()
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	// get iterator
	if a.RequiresIterator() || (reuse != nil && reuse.RequiresIterator()) {
		ait = a.Iterator()
		if reuse != nil {
			rit = reuse.Iterator()
		}
		useIter = true
	}
	return
}
