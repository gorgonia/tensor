package dense

import (
	"context"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

func (t *Dense[DT]) Trace() (retVal DT, err error) {
	if err := check(checkFlags(t.e, t)); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	if tracer, ok := t.e.Workhorse().(tensor.Tracer[DT]); ok {
		return tracer.Trace(context.Background(), t)
	}
	return retVal, errors.Errorf("Engine %T does not support Trace", t.e)
}

func (t *Dense[DT]) SVD(uv, full bool) (s, u, v *Dense[DT], err error) {
	if err := check(checkFlags(t.e, t)); err != nil {
		return s, u, v, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	svder, ok := t.e.Workhorse().(tensor.SVDer[DT, *Dense[DT]])
	if !ok {
		return nil, nil, nil, errors.Errorf(errors.EngineSupport, t.e, svder, errors.ThisFn())
	}
	return svder.SVD(context.Background(), t, uv, full)

	// sB, uB, vB, err := svder.SVD(context.Background(), t, uv, full)
	// if err != nil {
	// 	return nil, nil, nil, err
	// }
	// return sB.(*Dense[DT]), uB.(*Dense[DT]), vB.(*Dense[DT]), nil
}

func (t *Dense[DT]) Inner(u *Dense[DT]) (retVal DT, err error) {
	if err = check(checkFlags(t.e, t, u), checkCompatibleShape(t.Shape(), u.Shape())); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	var bla tensor.InnerProder[DT]
	var ok bool
	if bla, ok = t.e.Workhorse().(tensor.InnerProder[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, t.e, bla, errors.ThisFn())
	}
	return bla.Inner(context.Background(), t, u)
}

func (t *Dense[DT]) MatVecMul(u *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t), checkInnerProdDims(t, u), checkIsVector(u)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	var prepper tensor.SpecializedFuncOptHandler[DT, *Dense[DT]]
	var ok bool
	if prepper, ok = t.e.Workhorse().(tensor.SpecializedFuncOptHandler[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, prepper, errors.ThisFn())
	}
	expShape := elimInnermostOutermost(t.Shape(), u.Shape())
	var fo Option
	if retVal, fo, err = prepper.HandleFuncOptsSpecialized(t, expShape, opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}

	var bla tensor.BLA[DT]
	if bla, ok = t.e.Workhorse().(tensor.BLA[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, t.e, bla, errors.ThisFn())
	}
	var incr []DT
	if fo.Incr {
		incr = make([]DT, len(retVal.Data()))
		copy(incr, retVal.Data())
	}

	if err = bla.MatVecMul(fo.Ctx, t, u, retVal, incr); err != nil {
		return nil, err
	}
	return retVal, err
}

func (t *Dense[DT]) MatMul(u *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	e := t.e.Workhorse()
	if err = check(checkFlags(e, t, u), checkDims(2, t, u), checkInnerProdDims(t, u)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}

	expShape := elimInnermostOutermost(t.Shape(), u.Shape())
	retVal, fo, err := handleFuncOpts[DT, *Dense[DT]](e, t, expShape, opts...)
	if err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}

	var bla tensor.BLA[DT]
	var ok bool
	if bla, ok = e.(tensor.BLA[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, e, bla, errors.ThisFn())
	}
	var incr []DT
	if fo.Incr {
		incr = make([]DT, len(retVal.Data()))
		copy(incr, retVal.Data())
	}
	if err = bla.MatMul(fo.Ctx, t, u, retVal, incr); err != nil {
		return nil, err
	}
	return retVal, nil
}

func (t *Dense[DT]) Outer(u *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	if err = check(checkFlags(t.e, t)); err != nil {
		return nil, errors.Wrapf(err, errors.FailedSanity, errors.ThisFn())
	}
	var prepper tensor.SpecializedFuncOptHandler[DT, *Dense[DT]]
	var ok bool
	if prepper, ok = t.e.Workhorse().(tensor.SpecializedFuncOptHandler[DT, *Dense[DT]]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, prepper, errors.ThisFn())
	}
	var fo Option
	expShape := shapes.Shape{t.Shape().TotalSize(), u.Shape().TotalSize()}
	if retVal, fo, err = prepper.HandleFuncOptsSpecialized(t, expShape, opts...); err != nil {
		return nil, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}

	var bla tensor.BLA[DT]
	if bla, ok = t.e.Workhorse().(tensor.BLA[DT]); !ok {
		return nil, errors.Errorf(errors.EngineSupport, t.e, bla, errors.ThisFn())
	}
	var incr []DT
	if fo.Incr {
		incr = make([]DT, len(retVal.Data()))
		copy(incr, retVal.Data())
	}

	if err = bla.Outer(fo.Ctx, t, u, retVal, incr); err != nil {
		return nil, err
	}
	return retVal, nil
}

func (t *Dense[DT]) Norm(ord tensor.NormOrder, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	/*
		fo := ParseFuncOpts(opts...)
		axes := fo.Along
		ctx := fo.Ctx
		normEng, ok := t.e.Workhorse().(tensor.Normer[DT])
		if !ok {
			return nil, errors.Errorf(errors.EngineSupport, t.e, errors.ThisFn())
		}

		dims := t.Dims()
		if len(axes) == 0 {
			if ord.IsUnordered() || (ord.IsFrobenius() && dims == 2) || (ord == tensor.Norm(2) && dims == 1) {
				n := normEng.Norm2(ctx, t)
				return New[DT](WithEngine(t.e), FromScalar(n)), nil
			}
			axes = make([]int, dims)
			for i := range axes {
				axes[i] = i
			}
		}
		err = normEng.Norm(ctx, t, ord, axes)
		if err != nil {
			return nil, err
		}
	*/
	panic("NYI")
}

// TensorMul is for multiplying Tensors with more than 2 dimensions.
//
// The algorithm is conceptually simple (but tricky to get right):
//  1. Transpose and reshape the Tensors in such a way that both t and other are 2D matrices
//  2. Use DGEMM to multiply them
//  3. Reshape the results to be the new expected result
//
// This function is a Go implementation of Numpy's tensordot method. It simplifies a lot of what Numpy does.
func (t *Dense[DT]) TensorMul(other *Dense[DT], axesA, axesB []int, opts ...FuncOpt) (retVal *Dense[DT], err error) {
	ts := t.Shape()
	td := ts.Dims()

	os := other.Shape()
	od := os.Dims()

	na := len(axesA)
	nb := len(axesB)
	sameLength := na == nb
	if sameLength {
		for i := 0; i < na; i++ {
			dA, err1 := ts.Dim(axesA[i])
			dB, err2 := os.Dim(axesB[i])
			if err1 != nil || err2 != nil {
				sameLength = false
				break
			}
			if dA != dB {
				sameLength = false
				break
			}
			if axesA[i] < 0 {
				axesA[i] += td
			}

			if axesB[i] < 0 {
				axesB[i] += od
			}
		}
	}

	if !sameLength {
		err = errors.Errorf(errors.ShapeMismatch, ts, os)
		return
	}

	// handle shapes - we want the innermost dim of `t` to match the outermost dim of `other`.
	// so we will have to build up the correct shape and then transpose (and reshape)

	// To do that we first find the axes that are not included in t's shape.
	var notins []int
	for i := 0; i < td; i++ {
		notin := true
		for _, a := range axesA {
			if i == a {
				notin = false
				break
			}
		}
		if notin {
			notins = append(notins, i)
		}
	}

	newAxesA := internal.BorrowInts(len(notins) + len(axesA))
	defer internal.ReturnInts(newAxesA)
	newAxesA = newAxesA[:0]
	newAxesA = append(notins, axesA...)
	n2 := 1
	for _, a := range axesA {
		n2 *= ts[a]
	}

	newShapeT := shapes.Shape(internal.BorrowInts(2))
	defer internal.ReturnInts(newShapeT)
	newShapeT[0] = ts.TotalSize() / n2
	newShapeT[1] = n2

	retShape1 := internal.BorrowInts(len(ts))
	defer internal.ReturnInts(retShape1)
	retShape1 = retShape1[:0]
	for _, ni := range notins {
		retShape1 = append(retShape1, ts[ni])
	}

	// work on other now
	notins = notins[:0]
	for i := 0; i < od; i++ {
		notin := true
		for _, a := range axesB {
			if i == a {
				notin = false
				break
			}
		}
		if notin {
			notins = append(notins, i)
		}
	}

	newAxesB := internal.BorrowInts(len(notins) + len(axesB))
	defer internal.ReturnInts(newAxesB)
	newAxesB = newAxesB[:0]
	newAxesB = append(axesB, notins...)

	newShapeO := shapes.Shape(internal.BorrowInts(2))
	defer internal.ReturnInts(newShapeO)
	newShapeO[0] = n2
	newShapeO[1] = os.TotalSize() / n2

	retShape2 := internal.BorrowInts(len(ts))
	retShape2 = retShape2[:0]
	for _, ni := range notins {
		retShape2 = append(retShape2, os[ni])
	}

	var doT, doOther *Dense[DT]
	if doT, err = t.Transpose(newAxesA...); internal.HandleNoOp(err) != nil {
		return nil, err
	}
	if err = internal.HandleNoOp(doT.Reshape(newShapeT...)); err != nil {
		return nil, err
	}

	if doOther, err = other.Transpose(newAxesB...); internal.HandleNoOp(err) != nil {
		return nil, err
	}
	if err = internal.HandleNoOp(doOther.Reshape(newShapeO...)); err != nil {
		return nil, err
	}

	// magic happens here
	if retVal, err = doT.MatMul(doOther, opts...); err != nil {
		return nil, err
	}
	retShape := internal.BorrowInts(len(retShape1) + len(retShape2))
	retShape = retShape[:0]
	retShape = append(retShape, retShape1...)
	retShape = append(retShape, retShape2...)

	err = retVal.Reshape(retShape...)
	return
}

func (t *Dense[DT]) NPDot(other *Dense[DT], opts ...FuncOpt) (retVal *Dense[DT], err error) {
	panic("NYI")
}
