package tensor

import (
	"github.com/pkg/errors"
	"gorgonia.org/dtype"
	"gorgonia.org/tensor/internal/execution"
	"gorgonia.org/tensor/internal/storage"

	"gorgonia.org/vecf64"
)

func handleFuncOptsF64(expShape Shape, o DataOrder, opts ...FuncOpt) (reuse DenseTensor, safe, toReuse, incr bool, err error) {
	fo := ParseFuncOpts(opts...)

	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		var ok bool
		if reuse, ok = reuseT.(DenseTensor); !ok {
			returnOpOpt(fo)
			err = errors.Errorf("Cannot reuse a different type of Tensor in a *Dense-Scalar operation. Reuse is of %T", reuseT)
			return
		}
		if reuse.len() != expShape.TotalSize() && !expShape.IsScalar() {
			returnOpOpt(fo)
			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}

		if !incr && reuse != nil {
			reuse.setDataOrder(o)
			// err = reuse.reshape(expShape...)
		}

	}
	returnOpOpt(fo)
	return
}

func prepDataVSF64(a Tensor, b interface{}, reuse Tensor) (dataA *storage.Header, dataB float64, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	// get data
	dataA = a.hdr()
	switch bt := b.(type) {
	case float64:
		dataB = bt
	case *float64:
		dataB = *bt
	default:
		err = errors.Errorf("b is not a float64: %T", b)
		return
	}
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	if a.RequiresIterator() || (reuse != nil && reuse.RequiresIterator()) {
		ait = a.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
		useIter = true
	}
	return
}

func (e Float64Engine) checkThree(a, b Tensor, reuse Tensor) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	if !b.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, b)
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, reuse)
	}

	if a.Dtype() != Float64 {
		return errors.Errorf("Expected a to be of Float64. Got %v instead", a.Dtype())
	}
	if a.Dtype() != b.Dtype() || (reuse != nil && b.Dtype() != reuse.Dtype()) {
		return errors.Errorf("Expected a, b and reuse to have the same Dtype. Got %v, %v and %v instead", a.Dtype(), b.Dtype(), reuse.Dtype())
	}
	return nil
}

func (e Float64Engine) checkTwo(a Tensor, reuse Tensor) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	if reuse != nil && !reuse.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, reuse)
	}

	if a.Dtype() != Float64 {
		return errors.Errorf("Expected a to be of Float64. Got %v instead", a.Dtype())
	}

	if reuse != nil && reuse.Dtype() != a.Dtype() {
		return errors.Errorf("Expected reuse to be the same as a. Got %v instead", reuse.Dtype())
	}
	return nil
}

// Float64Engine is an execution engine that is optimized to only work with float64s. It assumes all data will are float64s.
//
// Use this engine only as form of optimization. You should probably be using the basic default engine for most cases.
type Float64Engine struct {
	StdEng
}

// makeArray allocates a slice for the array
func (e Float64Engine) makeArray(arr *array, t dtype.Dtype, size int) {
	if t != Float64 {
		panic("Float64Engine only creates float64s")
	}
	arr.Header.Raw = make([]byte, size*8)
	arr.t = t
}

func (e Float64Engine) FMA(a, x, y Tensor) (retVal Tensor, err error) {
	reuse := y
	if err = e.checkThree(a, x, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, _, err = prepDataVV(a, x, reuse); err != nil {
		return nil, errors.Wrap(err, "Float64Engine.FMA")
	}
	if useIter {
		err = execution.MulIterIncrF64(dataA.Float64s(), dataB.Float64s(), dataReuse.Float64s(), ait, bit, iit)
		retVal = reuse
		return
	}

	vecf64.IncrMul(dataA.Float64s(), dataB.Float64s(), dataReuse.Float64s())
	retVal = reuse
	return
}

func (e Float64Engine) FMAScalar(a Tensor, x interface{}, y Tensor) (retVal Tensor, err error) {
	reuse := y
	if err = e.checkTwo(a, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var ait, iit Iterator
	var dataTensor, dataReuse *storage.Header
	var scalar float64
	var useIter bool
	if dataTensor, scalar, dataReuse, ait, iit, useIter, err = prepDataVSF64(a, x, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "Float64Engine.FMAScalar")
	}
	if useIter {
		err = execution.MulIterIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s(), ait, iit)
		retVal = reuse
	}

	execution.MulIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s())
	retVal = reuse
	return
}

// Add performs a + b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e Float64Engine) Add(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.RequiresIterator() || b.RequiresIterator() {
		return e.StdEng.Add(a, b, opts...)
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = handleFuncOptsF64(a.Shape(), a.DataOrder(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = e.checkThree(a, b, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var hdrA, hdrB, hdrReuse *storage.Header
	var dataA, dataB, dataReuse []float64

	if hdrA, hdrB, hdrReuse, _, _, _, _, _, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "Float64Engine.Add")
	}
	dataA = hdrA.Float64s()
	dataB = hdrB.Float64s()
	if hdrReuse != nil {
		dataReuse = hdrReuse.Float64s()
	}

	switch {
	case incr:
		vecf64.IncrAdd(dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		copy(dataReuse, dataA)
		vecf64.Add(dataReuse, dataB)
		retVal = reuse
	case !safe:
		vecf64.Add(dataA, dataB)
		retVal = a
	default:
		ret, ok := a.Clone().(float64ser)
		if !ok {
			return nil, errors.Errorf("Unable to get the Float64 data from `a`, of %T", a)
		}
		vecf64.Add(ret.Float64s(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e Float64Engine) Inner(a, b Tensor) (retVal float64, err error) {
	var A, B []float64
	var AD, BD *Dense
	var ok bool

	if AD, ok = a.(*Dense); !ok {
		return 0, errors.Errorf("a is not a *Dense")
	}
	if BD, ok = b.(*Dense); !ok {
		return 0, errors.Errorf("b is not a *Dense")
	}

	A = AD.Float64s()
	B = BD.Float64s()
	retVal = whichblas.Ddot(len(A), A, 1, B, 1)
	return
}
