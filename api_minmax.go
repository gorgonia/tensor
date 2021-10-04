package tensor

import "github.com/pkg/errors"

func MinBetween(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var minbetweener MinBetweener
	var oe standardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe = at.standardEngine()
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor addition
				if oe != nil {
					return oe.MinBetween(at, bt, opts...)
				}
				if oe = bt.standardEngine(); oe != nil {
					return oe.MinBetween(at, bt, opts...)
				}
				if minbetweener, ok = at.Engine().(MinBetweener); ok {
					return minbetweener.MinBetween(at, bt, opts...)
				}
				if minbetweener, ok = bt.Engine().(MinBetweener); ok {
					return minbetweener.MinBetween(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support MinBetween")

			} else { // at least one of the operands is a scalar
				var leftTensor bool
				if !bt.Shape().IsScalar() {
					leftTensor = false // a Scalar-Tensor * b Tensor
					tmp := at
					at = bt
					bt = tmp
				} else {
					leftTensor = true // a Tensor * b Scalar-Tensor
				}

				if oe != nil {
					return oe.MinBetweenScalar(at, bt, leftTensor, opts...)
				}
				if oe = bt.standardEngine(); oe != nil {
					return oe.MinBetweenScalar(at, bt, leftTensor, opts...)
				}
				if minbetweener, ok = at.Engine().(MinBetweener); ok {
					return minbetweener.MinBetweenScalar(at, bt, leftTensor, opts...)
				}
				if minbetweener, ok = bt.Engine().(MinBetweener); ok {
					return minbetweener.MinBetweenScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support MinBetween")
			}

		default:
			if oe != nil {
				return oe.MinBetweenScalar(at, bt, true, opts...)
			}
			if minbetweener, ok = at.Engine().(MinBetweener); ok {
				return minbetweener.MinBetweenScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support MinBetween")
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if oe = bt.standardEngine(); oe != nil {
				return oe.MinBetweenScalar(bt, at, false, opts...)
			}
			if minbetweener, ok = bt.Engine().(MinBetweener); ok {
				return minbetweener.MinBetweenScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support MinBetween")
		default:
			return nil, errors.Errorf("Cannot perform MinBetween of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

func MaxBetween(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var maxbetweener MaxBetweener
	var oe standardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe = at.standardEngine()
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor addition
				if oe != nil {
					return oe.MaxBetween(at, bt, opts...)
				}
				if oe = bt.standardEngine(); oe != nil {
					return oe.MaxBetween(at, bt, opts...)
				}
				if maxbetweener, ok = at.Engine().(MaxBetweener); ok {
					return maxbetweener.MaxBetween(at, bt, opts...)
				}
				if maxbetweener, ok = bt.Engine().(MaxBetweener); ok {
					return maxbetweener.MaxBetween(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support MaxBetween")

			} else { // at least one of the operands is a scalar
				var leftTensor bool
				if !bt.Shape().IsScalar() {
					leftTensor = false // a Scalar-Tensor * b Tensor
					tmp := at
					at = bt
					bt = tmp
				} else {
					leftTensor = true // a Tensor * b Scalar-Tensor
				}

				if oe != nil {
					return oe.MaxBetweenScalar(at, bt, leftTensor, opts...)
				}
				if oe = bt.standardEngine(); oe != nil {
					return oe.MaxBetweenScalar(at, bt, leftTensor, opts...)
				}
				if maxbetweener, ok = at.Engine().(MaxBetweener); ok {
					return maxbetweener.MaxBetweenScalar(at, bt, leftTensor, opts...)
				}
				if maxbetweener, ok = bt.Engine().(MaxBetweener); ok {
					return maxbetweener.MaxBetweenScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support MaxBetween")
			}

		default:
			if oe != nil {
				return oe.MaxBetweenScalar(at, bt, true, opts...)
			}
			if maxbetweener, ok = at.Engine().(MaxBetweener); ok {
				return maxbetweener.MaxBetweenScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support MaxBetween")
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if oe = bt.standardEngine(); oe != nil {
				return oe.MaxBetweenScalar(bt, at, false, opts...)
			}
			if maxbetweener, ok = bt.Engine().(MaxBetweener); ok {
				return maxbetweener.MaxBetweenScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support MaxBetween")
		default:
			return nil, errors.Errorf("Cannot perform MaxBetween of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}
