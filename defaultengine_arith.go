// Code generated by genlib2. DO NOT EDIT.

package tensor

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"
)

// Add performs a + b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) Add(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Add failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Add")
	}
	if useIter {
		switch {
		case incr:
			err = e.E.AddIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.AddIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.AddIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			if swap {
				retVal = b.Clone().(Tensor)
			} else {
				retVal = a.Clone().(Tensor)
			}
			err = e.E.AddIter(typ, retVal.hdr(), dataB, ait, bit)
		}

		return
	}
	switch {
	case incr:
		err = e.E.AddIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Add(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Add(typ, dataA, dataB)
		retVal = a
	default:
		if swap {
			retVal = b.Clone().(Tensor)
		} else {
			retVal = a.Clone().(Tensor)
		}
		err = e.E.Add(typ, retVal.hdr(), dataB)
	}

	return
}

// Sub performs a - b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) Sub(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Sub failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Sub")
	}
	if useIter {
		switch {
		case incr:
			err = e.E.SubIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.SubIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.SubIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			if swap {
				retVal = b.Clone().(Tensor)
			} else {
				retVal = a.Clone().(Tensor)
			}
			err = e.E.SubIter(typ, retVal.hdr(), dataB, ait, bit)
		}

		return
	}
	switch {
	case incr:
		err = e.E.SubIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Sub(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Sub(typ, dataA, dataB)
		retVal = a
	default:
		if swap {
			retVal = b.Clone().(Tensor)
		} else {
			retVal = a.Clone().(Tensor)
		}
		err = e.E.Sub(typ, retVal.hdr(), dataB)
	}

	return
}

// Mul performs a × b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) Mul(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Mul failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Mul")
	}
	if useIter {
		switch {
		case incr:
			err = e.E.MulIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.MulIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.MulIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			if swap {
				retVal = b.Clone().(Tensor)
			} else {
				retVal = a.Clone().(Tensor)
			}
			err = e.E.MulIter(typ, retVal.hdr(), dataB, ait, bit)
		}

		return
	}
	switch {
	case incr:
		err = e.E.MulIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mul(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Mul(typ, dataA, dataB)
		retVal = a
	default:
		if swap {
			retVal = b.Clone().(Tensor)
		} else {
			retVal = a.Clone().(Tensor)
		}
		err = e.E.Mul(typ, retVal.hdr(), dataB)
	}

	return
}

// Div performs a ÷ b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) Div(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Div failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Div")
	}
	if useIter {
		switch {
		case incr:
			err = e.E.DivIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.DivIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.DivIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			if swap {
				retVal = b.Clone().(Tensor)
			} else {
				retVal = a.Clone().(Tensor)
			}
			err = e.E.DivIter(typ, retVal.hdr(), dataB, ait, bit)
		}

		return
	}
	switch {
	case incr:
		err = e.E.DivIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Div(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Div(typ, dataA, dataB)
		retVal = a
	default:
		if swap {
			retVal = b.Clone().(Tensor)
		} else {
			retVal = a.Clone().(Tensor)
		}
		err = e.E.Div(typ, retVal.hdr(), dataB)
	}

	return
}

// Pow performs a ^ b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) Pow(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Pow failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Pow")
	}
	if useIter {
		switch {
		case incr:
			err = e.E.PowIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.PowIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.PowIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			if swap {
				retVal = b.Clone().(Tensor)
			} else {
				retVal = a.Clone().(Tensor)
			}
			err = e.E.PowIter(typ, retVal.hdr(), dataB, ait, bit)
		}

		return
	}
	switch {
	case incr:
		err = e.E.PowIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Pow(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Pow(typ, dataA, dataB)
		retVal = a
	default:
		if swap {
			retVal = b.Clone().(Tensor)
		} else {
			retVal = a.Clone().(Tensor)
		}
		err = e.E.Pow(typ, retVal.hdr(), dataB)
	}

	return
}

// Mod performs a % b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) Mod(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Mod failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), a.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Mod")
	}
	if useIter {
		switch {
		case incr:
			err = e.E.ModIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.ModIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.ModIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			if swap {
				retVal = b.Clone().(Tensor)
			} else {
				retVal = a.Clone().(Tensor)
			}
			err = e.E.ModIter(typ, retVal.hdr(), dataB, ait, bit)
		}

		return
	}
	switch {
	case incr:
		err = e.E.ModIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mod(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Mod(typ, dataA, dataB)
		retVal = a
	default:
		if swap {
			retVal = b.Clone().(Tensor)
		} else {
			retVal = a.Clone().(Tensor)
		}
		err = e.E.Mod(typ, retVal.hdr(), dataB)
	}

	return
}

// AddScalar performs t + s elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in s.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) AddScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Add failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(t.Shape(), t.Dtype(), t.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse, scalarHeader *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Add")
		}
		scalarHeader = dataB
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Add")
		}
		scalarHeader = dataA
	}

	if useIter {
		switch {
		case incr:
			err = e.E.AddIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.AddIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			iit.Reset()
			bit.Reset()
			err = e.E.AddIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.AddIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			retVal = a.Clone().(Tensor)
			if leftTensor {
				err = e.E.AddIter(typ, retVal.hdr(), dataB, ait, bit)
			} else {
				err = e.E.AddIter(typ, dataA, retVal.hdr(), ait, bit)
			}
		}
		returnHeader(scalarHeader)
		return
	}
	switch {
	case incr:
		err = e.E.AddIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Add(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Add(typ, dataA, dataReuse)
		if t.Shape().IsScalarEquiv() {
			storage.Copy(typ, dataReuse, dataA)
		}
		retVal = reuse
	case !safe:
		err = e.E.Add(typ, dataA, dataB)
		if t.Shape().IsScalarEquiv() && !leftTensor {
			storage.Copy(typ, dataB, dataA)
		}
		retVal = a
	default:
		retVal = a.Clone().(Tensor)
		if !leftTensor {
			storage.Fill(typ, retVal.hdr(), dataA)
		}
		err = e.E.Add(typ, retVal.hdr(), dataB)
	}
	returnHeader(scalarHeader)
	return
}

// SubScalar performs t - s elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in s.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) SubScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Sub failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(t.Shape(), t.Dtype(), t.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse, scalarHeader *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Sub")
		}
		scalarHeader = dataB
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Sub")
		}
		scalarHeader = dataA
	}

	if useIter {
		switch {
		case incr:
			err = e.E.SubIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.SubIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			iit.Reset()
			bit.Reset()
			err = e.E.SubIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.SubIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			retVal = a.Clone().(Tensor)
			if leftTensor {
				err = e.E.SubIter(typ, retVal.hdr(), dataB, ait, bit)
			} else {
				err = e.E.SubIter(typ, dataA, retVal.hdr(), ait, bit)
			}
		}
		returnHeader(scalarHeader)
		return
	}
	switch {
	case incr:
		err = e.E.SubIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Sub(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Sub(typ, dataA, dataReuse)
		if t.Shape().IsScalarEquiv() {
			storage.Copy(typ, dataReuse, dataA)
		}
		retVal = reuse
	case !safe:
		err = e.E.Sub(typ, dataA, dataB)
		if t.Shape().IsScalarEquiv() && !leftTensor {
			storage.Copy(typ, dataB, dataA)
		}
		retVal = a
	default:
		retVal = a.Clone().(Tensor)
		if !leftTensor {
			storage.Fill(typ, retVal.hdr(), dataA)
		}
		err = e.E.Sub(typ, retVal.hdr(), dataB)
	}
	returnHeader(scalarHeader)
	return
}

// MulScalar performs t × s elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in s.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) MulScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Mul failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(t.Shape(), t.Dtype(), t.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse, scalarHeader *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Mul")
		}
		scalarHeader = dataB
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Mul")
		}
		scalarHeader = dataA
	}

	if useIter {
		switch {
		case incr:
			err = e.E.MulIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.MulIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			iit.Reset()
			bit.Reset()
			err = e.E.MulIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.MulIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			retVal = a.Clone().(Tensor)
			if leftTensor {
				err = e.E.MulIter(typ, retVal.hdr(), dataB, ait, bit)
			} else {
				err = e.E.MulIter(typ, dataA, retVal.hdr(), ait, bit)
			}
		}
		returnHeader(scalarHeader)
		return
	}
	switch {
	case incr:
		err = e.E.MulIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mul(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Mul(typ, dataA, dataReuse)
		if t.Shape().IsScalarEquiv() {
			storage.Copy(typ, dataReuse, dataA)
		}
		retVal = reuse
	case !safe:
		err = e.E.Mul(typ, dataA, dataB)
		if t.Shape().IsScalarEquiv() && !leftTensor {
			storage.Copy(typ, dataB, dataA)
		}
		retVal = a
	default:
		retVal = a.Clone().(Tensor)
		if !leftTensor {
			storage.Fill(typ, retVal.hdr(), dataA)
		}
		err = e.E.Mul(typ, retVal.hdr(), dataB)
	}
	returnHeader(scalarHeader)
	return
}

// DivScalar performs t ÷ s elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in s.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) DivScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Div failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(t.Shape(), t.Dtype(), t.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse, scalarHeader *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Div")
		}
		scalarHeader = dataB
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Div")
		}
		scalarHeader = dataA
	}

	if useIter {
		switch {
		case incr:
			err = e.E.DivIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.DivIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			iit.Reset()
			bit.Reset()
			err = e.E.DivIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.DivIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			retVal = a.Clone().(Tensor)
			if leftTensor {
				err = e.E.DivIter(typ, retVal.hdr(), dataB, ait, bit)
			} else {
				err = e.E.DivIter(typ, dataA, retVal.hdr(), ait, bit)
			}
		}
		returnHeader(scalarHeader)
		return
	}
	switch {
	case incr:
		err = e.E.DivIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Div(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Div(typ, dataA, dataReuse)
		if t.Shape().IsScalarEquiv() {
			storage.Copy(typ, dataReuse, dataA)
		}
		retVal = reuse
	case !safe:
		err = e.E.Div(typ, dataA, dataB)
		if t.Shape().IsScalarEquiv() && !leftTensor {
			storage.Copy(typ, dataB, dataA)
		}
		retVal = a
	default:
		retVal = a.Clone().(Tensor)
		if !leftTensor {
			storage.Fill(typ, retVal.hdr(), dataA)
		}
		err = e.E.Div(typ, retVal.hdr(), dataB)
	}
	returnHeader(scalarHeader)
	return
}

// PowScalar performs t ^ s elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in s.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) PowScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Pow failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(t.Shape(), t.Dtype(), t.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse, scalarHeader *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Pow")
		}
		scalarHeader = dataB
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Pow")
		}
		scalarHeader = dataA
	}

	if useIter {
		switch {
		case incr:
			err = e.E.PowIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.PowIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			iit.Reset()
			bit.Reset()
			err = e.E.PowIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.PowIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			retVal = a.Clone().(Tensor)
			if leftTensor {
				err = e.E.PowIter(typ, retVal.hdr(), dataB, ait, bit)
			} else {
				err = e.E.PowIter(typ, dataA, retVal.hdr(), ait, bit)
			}
		}
		returnHeader(scalarHeader)
		return
	}
	switch {
	case incr:
		err = e.E.PowIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Pow(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Pow(typ, dataA, dataReuse)
		if t.Shape().IsScalarEquiv() {
			storage.Copy(typ, dataReuse, dataA)
		}
		retVal = reuse
	case !safe:
		err = e.E.Pow(typ, dataA, dataB)
		if t.Shape().IsScalarEquiv() && !leftTensor {
			storage.Copy(typ, dataB, dataA)
		}
		retVal = a
	default:
		retVal = a.Clone().(Tensor)
		if !leftTensor {
			storage.Fill(typ, retVal.hdr(), dataA)
		}
		err = e.E.Pow(typ, retVal.hdr(), dataB)
	}
	returnHeader(scalarHeader)
	return
}

// ModScalar performs t % s elementwise. The leftTensor parameter indicates if the tensor is the left operand. Only scalar types are accepted in s.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e StdEng) ModScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, numberTypes); err != nil {
		return nil, errors.Wrapf(err, "Mod failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(t.Shape(), t.Dtype(), t.DataOrder(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse, scalarHeader *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Mod")
		}
		scalarHeader = dataB
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Mod")
		}
		scalarHeader = dataA
	}

	if useIter {
		switch {
		case incr:
			err = e.E.ModIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.ModIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			iit.Reset()
			bit.Reset()
			err = e.E.ModIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.ModIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			retVal = a.Clone().(Tensor)
			if leftTensor {
				err = e.E.ModIter(typ, retVal.hdr(), dataB, ait, bit)
			} else {
				err = e.E.ModIter(typ, dataA, retVal.hdr(), ait, bit)
			}
		}
		returnHeader(scalarHeader)
		return
	}
	switch {
	case incr:
		err = e.E.ModIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mod(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Mod(typ, dataA, dataReuse)
		if t.Shape().IsScalarEquiv() {
			storage.Copy(typ, dataReuse, dataA)
		}
		retVal = reuse
	case !safe:
		err = e.E.Mod(typ, dataA, dataB)
		if t.Shape().IsScalarEquiv() && !leftTensor {
			storage.Copy(typ, dataB, dataA)
		}
		retVal = a
	default:
		retVal = a.Clone().(Tensor)
		if !leftTensor {
			storage.Fill(typ, retVal.hdr(), dataA)
		}
		err = e.E.Mod(typ, retVal.hdr(), dataB)
	}
	returnHeader(scalarHeader)
	return
}
