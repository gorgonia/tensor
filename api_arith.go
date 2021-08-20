package tensor

import (
	"github.com/pkg/errors"
)

// exported API for arithmetics and the stupidly crazy amount of overloaded semantics

// Add performs a pointwise a+b. a and b can either be float64 or Tensor
//
// If both operands are Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//

// Add performs elementwise addition on the Tensor(s). These operations are supported:
//		Add(*Dense, scalar)
//		Add(scalar, *Dense)
//		Add(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Add(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var adder Adder
	var oe StandardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe, _ = at.Engine().(StandardEngine)
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor addition
				if oe != nil {
					return oe.Add(at, bt, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.Add(at, bt, opts...)
				}
				if adder, ok = at.Engine().(Adder); ok {
					return adder.Add(at, bt, opts...)
				}
				if adder, ok = bt.Engine().(Adder); ok {
					return adder.Add(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Add")

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
					return oe.AddScalar(at, bt, leftTensor, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.AddScalar(at, bt, leftTensor, opts...)
				}
				if adder, ok = at.Engine().(Adder); ok {
					return adder.AddScalar(at, bt, leftTensor, opts...)
				}
				if adder, ok = bt.Engine().(Adder); ok {
					return adder.AddScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Add")
			}

		default:
			if oe != nil {
				return oe.AddScalar(at, bt, true, opts...)
			}
			if adder, ok = at.Engine().(Adder); ok {
				return adder.AddScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support Add")
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if oe, ok = bt.Engine().(StandardEngine); ok {
				return oe.AddScalar(bt, at, false, opts...)
			}
			if adder, ok = bt.Engine().(Adder); ok {
				return adder.AddScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support Add")
		default:
			return nil, errors.Errorf("Cannot perform Add of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

// Sub performs elementwise subtraction on the Tensor(s). These operations are supported:
//		Sub(*Dense, scalar)
//		Sub(scalar, *Dense)
//		Sub(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Sub(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var suber Suber
	var oe StandardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe, _ = at.Engine().(StandardEngine)
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor substraction
				if oe != nil {
					return oe.Sub(at, bt, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.Sub(at, bt, opts...)
				}
				if suber, ok = at.Engine().(Suber); ok {
					return suber.Sub(at, bt, opts...)
				}
				if suber, ok = bt.Engine().(Suber); ok {
					return suber.Sub(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Sub")

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
					return oe.SubScalar(at, bt, leftTensor, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.SubScalar(at, bt, leftTensor, opts...)
				}
				if suber, ok = at.Engine().(Suber); ok {
					return suber.SubScalar(at, bt, leftTensor, opts...)
				}
				if suber, ok = bt.Engine().(Suber); ok {
					return suber.SubScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Sub")
			}

		default:
			if oe != nil {
				return oe.SubScalar(at, bt, true, opts...)
			}
			if suber, ok = at.Engine().(Suber); ok {
				return suber.SubScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support Sub")
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if oe, ok = bt.Engine().(StandardEngine); ok {
				return oe.SubScalar(bt, at, false, opts...)
			}
			if suber, ok = bt.Engine().(Suber); ok {
				return suber.SubScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support Sub")
		default:
			return nil, errors.Errorf("Cannot perform Sub of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

// Mul performs elementwise multiplication on the Tensor(s). These operations are supported:
//		Mul(*Dense, scalar)
//		Mul(scalar, *Dense)
//		Mul(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Mul(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var muler Muler
	var oe StandardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe, _ = at.Engine().(StandardEngine)
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor multiplication
				if oe != nil {
					return oe.Mul(at, bt, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.Mul(at, bt, opts...)
				}
				if muler, ok = at.Engine().(Muler); ok {
					return muler.Mul(at, bt, opts...)
				}
				if muler, ok = bt.Engine().(Muler); ok {
					return muler.Mul(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Mul")

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
					return oe.MulScalar(at, bt, leftTensor, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.MulScalar(at, bt, leftTensor, opts...)
				}
				if muler, ok = at.Engine().(Muler); ok {
					return muler.MulScalar(at, bt, leftTensor, opts...)
				}
				if muler, ok = bt.Engine().(Muler); ok {
					return muler.MulScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Mul")
			}

		default: // a Tensor * b interface
			if oe != nil {
				return oe.MulScalar(at, bt, true, opts...)
			}
			if muler, ok = at.Engine().(Muler); ok {
				return muler.MulScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support Mul")
		}

	default:
		switch bt := b.(type) {
		case Tensor: // b Tensor * a interface
			if oe, ok = bt.Engine().(StandardEngine); ok {
				return oe.MulScalar(bt, at, false, opts...)
			}
			if muler, ok = bt.Engine().(Muler); ok {
				return muler.MulScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support Mul")

		default: // b interface * a interface
			return nil, errors.Errorf("Cannot perform Mul of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

// Div performs elementwise division on the Tensor(s). These operations are supported:
//		Div(*Dense, scalar)
//		Div(scalar, *Dense)
//		Div(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Div(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var diver Diver
	var oe StandardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe, _ = at.Engine().(StandardEngine)
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor division
				if oe != nil {
					return oe.Div(at, bt, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.Div(at, bt, opts...)
				}
				if diver, ok = at.Engine().(Diver); ok {
					return diver.Div(at, bt, opts...)
				}
				if diver, ok = bt.Engine().(Diver); ok {
					return diver.Div(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Div")

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
					return oe.DivScalar(at, bt, leftTensor, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.DivScalar(at, bt, leftTensor, opts...)
				}
				if diver, ok = at.Engine().(Diver); ok {
					return diver.DivScalar(at, bt, leftTensor, opts...)
				}
				if diver, ok = bt.Engine().(Diver); ok {
					return diver.DivScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Div")
			}

		default:
			if oe != nil {
				return oe.DivScalar(at, bt, true, opts...)
			}
			if diver, ok = at.Engine().(Diver); ok {
				return diver.DivScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support Div")
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if oe, ok = bt.Engine().(StandardEngine); ok {
				return oe.DivScalar(bt, at, false, opts...)
			}
			if diver, ok = bt.Engine().(Diver); ok {
				return diver.DivScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support Div")
		default:
			return nil, errors.Errorf("Cannot perform Div of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

// Pow performs elementwise exponentiation on the Tensor(s). These operations are supported:
//		Pow(*Dense, scalar)
//		Pow(scalar, *Dense)
//		Pow(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Pow(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var power Power
	var oe StandardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe, _ = at.Engine().(StandardEngine)
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor exponentiation
				if oe != nil {
					return oe.Pow(at, bt, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.Pow(at, bt, opts...)
				}
				if power, ok = at.Engine().(Power); ok {
					return power.Pow(at, bt, opts...)
				}
				if power, ok = bt.Engine().(Power); ok {
					return power.Pow(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Pow")

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
					return oe.PowScalar(at, bt, leftTensor, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.PowScalar(at, bt, leftTensor, opts...)
				}
				if power, ok = at.Engine().(Power); ok {
					return power.PowScalar(at, bt, leftTensor, opts...)
				}
				if power, ok = bt.Engine().(Power); ok {
					return power.PowScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Pow")
			}

		default:
			if oe != nil {
				return oe.PowScalar(at, bt, true, opts...)
			}
			if power, ok = at.Engine().(Power); ok {
				return power.PowScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support Pow")
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if oe, ok = bt.Engine().(StandardEngine); ok {
				return oe.PowScalar(bt, at, false, opts...)
			}
			if power, ok = bt.Engine().(Power); ok {
				return power.PowScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support Pow")
		default:
			return nil, errors.Errorf("Cannot perform Pow of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

// Mod performs elementwise modulo on the Tensor(s). These operations are supported:
//		Mod(*Dense, scalar)
//		Mod(scalar, *Dense)
//		Mod(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Mod(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var moder Moder
	var oe StandardEngine
	var ok bool
	switch at := a.(type) {
	case Tensor:
		oe, _ = at.Engine().(StandardEngine)
		switch bt := b.(type) {
		case Tensor:
			if !bt.Shape().IsScalar() && !at.Shape().IsScalar() { // non-scalar Tensor modulo
				if oe != nil {
					return oe.Mod(at, bt, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.Mod(at, bt, opts...)
				}
				if moder, ok = at.Engine().(Moder); ok {
					return moder.Mod(at, bt, opts...)
				}
				if moder, ok = bt.Engine().(Moder); ok {
					return moder.Mod(at, bt, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Mod")

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
					return oe.ModScalar(at, bt, leftTensor, opts...)
				}
				if oe, ok = bt.Engine().(StandardEngine); ok {
					return oe.ModScalar(at, bt, leftTensor, opts...)
				}
				if moder, ok = at.Engine().(Moder); ok {
					return moder.ModScalar(at, bt, leftTensor, opts...)
				}
				if moder, ok = bt.Engine().(Moder); ok {
					return moder.ModScalar(at, bt, leftTensor, opts...)
				}
				return nil, errors.New("Neither engines of either operand support Mod")
			}

		default:
			if oe != nil {
				return oe.ModScalar(at, bt, true, opts...)
			}
			if moder, ok = at.Engine().(Moder); ok {
				return moder.ModScalar(at, bt, true, opts...)
			}
			return nil, errors.New("Operand A's engine does not support Mod")
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if oe, ok = bt.Engine().(StandardEngine); ok {
				return oe.ModScalar(bt, at, false, opts...)
			}
			if moder, ok = bt.Engine().(Moder); ok {
				return moder.ModScalar(bt, at, false, opts...)
			}
			return nil, errors.New("Operand B's engine does not support Mod")
		default:
			return nil, errors.Errorf("Cannot perform Mod of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

// Dot is a highly opinionated API for performing dot product operations on two *Denses, a and b.
// This function is opinionated with regard to the vector operations because of how it treats operations with vectors.
// Vectors in this package comes in two flavours - column or row vectors. Column vectors have shape (x, 1), while row vectors have shape (1, x).
//
// As such, it is easy to assume that performing a linalg operation on vectors would follow the same rules (i.e shapes have to be aligned for things to work).
// For the most part in this package, this is true. This function is one of the few notable exceptions.
//
// Here I give three specific examples of how the expectations of vector operations will differ.
// 		Given two vectors, a, b with shapes (4, 1) and (4, 1), Dot() will perform an inner product as if the shapes were (1, 4) and (4, 1). This will result in a scalar value
// 		Given matrix A and vector b with shapes (2, 4) and (1, 4), Dot() will perform a matrix-vector multiplication as if the shapes were (2,4) and (4,1). This will result in a column vector with shape (2,1)
//		Given vector a and matrix B with shapes (3, 1) and (3, 2), Dot() will perform a matrix-vector multiplication as if it were Bᵀ * a
//
// The main reason why this opinionated route was taken was due to the author's familiarity with NumPy, and general laziness in translating existing machine learning algorithms
// to fit the API of the package.
func Dot(x, y Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if xdottir, ok := x.Engine().(Dotter); ok {
		return xdottir.Dot(x, y, opts...)
	}
	if ydottir, ok := y.Engine().(Dotter); ok {
		return ydottir.Dot(x, y, opts...)
	}
	return nil, errors.New("Neither x's nor y's engines support Dot")
}

// FMA performs Y = A * X + Y.
func FMA(a Tensor, x interface{}, y Tensor) (retVal Tensor, err error) {
	var fm FMAer

	if xTensor, ok := x.(Tensor); ok {
		for _, T := range [3]Tensor{a, xTensor, y} {
			e := T.Engine()
			ctx := ctxFromEngine(e)
			fm, ok = e.(FMAer)
			if ok {
				return fm.FMA(ctx, a, xTensor, y)
			}
		}
	} else {
		for _, T := range [2]Tensor{a, y} {
			e := T.Engine()
			ctx := ctxFromEngine(e)
			fm, ok = e.(FMAer)
			if ok {
				return fm.FMAScalar(ctx, a, x, y)
			}
		}
	}

	return Mul(a, x, WithIncr(y))
}

// MatMul performs matrix-matrix multiplication between two Tensors
func MatMul(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}
	ad, aok := a.(*Dense)
	_, bok := b.(*Dense)
	if aok && bok {
		// fast path
		return ad.MatMul(b, opts...)
	}

	// check that both are matrices
	if !a.Shape().IsMatrix() || !b.Shape().IsMatrix() {
		err = errors.Errorf("MatMul requires both operands to be matrices. Got t's shape: %v, other's shape: %v", a.Shape(), b.Shape())
		return
	}

	// checks that t is mxk matrix
	var m, n, k int
	m = a.Shape()[0]
	k = a.Shape()[1]
	n = b.Shape()[1]

	// check shape
	if k != b.Shape()[0] {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return
	}

	// check whether retVal has the same size as the resulting matrix would be: mxn
	expectedShape := Shape{m, n}

	eng := a.Engine()
	mm, ok := eng.(MatMuler)
	if !ok {
		eng = b.Engine()
		mm, ok = eng.(MatMuler)
	}
	if !ok {
		return nil, errors.Errorf("Neither a or b have an engine that is a MatMuler. a: %T, b: %T", a.Engine(), b.Engine())
	}

	var reuse Tensor
	fo := ParseFuncOpts(opts...)
	defer returnOpOpt(fo)
	ctx := fo.Context()
	reuse = fo.Reuse()
	if reuse == nil {
		return nil, errors.Errorf("MatMul requires passing in of a reuse Tensor for now.")
	}

	if err := checkFixShape(reuse, expectedShape); err != nil {
		return nil, errors.Wrapf(err, opFail, "MatMul")
	}
	if err = mm.MatMul(ctx, a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "MatMul")
	}

	incr := fo.Incr()
	if incr != nil {
		return Add(incr, reuse, UseUnsafe())
	}
	return reuse, nil

}

// MatVecMul performs matrix-vector multiplication between two Tensors. `a` is expected to be a matrix, and `b` is expected to be a vector
func MatVecMul(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.MatVecMul(bt, opts...)
	}
	panic("Unreachable")
}

// Inner finds the inner products of two vector Tensors. Both arguments to the functions are eexpected to be vectors.
func Inner(a, b Tensor, opts ...FuncOpt) (retVal interface{}, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.Inner(bt, opts...)
	}
	panic("Unreachable")
}

// Outer performs the outer product of two vector Tensors. Both arguments to the functions are expected to be vectors.
func Outer(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.Outer(bt, opts...)
	}
	panic("Unreachable")
}

// Contract performs a contraction of given tensors along given axes
func Contract(a, b Tensor, aAxes, bAxes []int) (retVal Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.TensorMul(bt, aAxes, bAxes)

	default:
		panic("Unreachable")
	}
}
