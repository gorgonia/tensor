package dense

import (
	"context"
	"math"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
	stdeng "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
	"gorgonia.org/vecf64"
)

var (
	_ tensor.Adder[float64]       = StdFloat64Engine[*Dense[float64]]{}
	_ tensor.Reducer[float64]     = StdFloat64Engine[*Dense[float64]]{}
	_ tensor.Argmethoder[float64] = StdFloat64Engine[*Dense[float64]]{}
	_ tensor.Scatterer[float64]   = StdFloat64Engine[*Dense[float64]]{}
	_ tensor.Ord[float64]         = StdFloat64Engine[*Dense[float64]]{}
)

// type StdFloat64Engine[T tensor.Tensor[float64, T]] struct {
type StdFloat64Engine[T tensor.Basic[float64]] struct {
	stdeng.StdOrderedNumEngine[float64, T]
	blas gonum.Implementation // the default BLAS implementation uses gonum's native implementation
}

// Workhorse returns the engine that will actually do all the work (in this case, itself).
func (e StdFloat64Engine[T]) Workhorse() Engine { return e }

func (e StdFloat64Engine[T]) BasicEng() Engine {
	return StdFloat64Engine[tensor.Basic[float64]]{}
}

func (e StdFloat64Engine[T]) SVD(ctx context.Context, a T, uv, full bool) (s, u, v T, err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return
	}

	var ok bool
	// if err = e.checkAccessible(a); err != nil {
	// 	return s, u, v, errors.Wrapf(err, "opFail %v", "SVD")
	// }

	if !tensor.IsMatrix(a) {
		return s, u, v, errors.Errorf(errors.DimMismatch, 2, a.Dims())
	}

	var m *mat.Dense
	var svd mat.SVD

	if m, err = ToMat64[float64](a, UseUnsafe()); err != nil {
		return
	}

	switch {
	case full && uv:
		ok = svd.Factorize(m, mat.SVDFull)
	case !full && uv:
		ok = svd.Factorize(m, mat.SVDThin)
	case full && !uv:
		// illogical state - if you specify "full", you WANT the UV matrices
		// error
		err = errors.Errorf("SVD requires computation of `u` and `v` matrices if `full` was specified.")
		return
	default:
		// by default, we return only the singular values
		ok = svd.Factorize(m, mat.SVDNone)
	}

	if !ok {
		// error
		err = errors.Errorf("Unable to compute SVD")
		return
	}

	// extract values

	// mk is a maker of T. Specifically, what we want to use is the FromDense method of T
	// Now, we could use an interface, but since T is already constrained to a DensTensor,
	// which has a `FromDense` constraint, we can just directly use T.
	var z T
	mk, ok := any(z).(FromDenser[float64, T])
	if !ok {
		return s, u, v, errors.Errorf("Unable to build a constructor from %T", z)
	}

	var um, vm mat.Dense
	s = mk.FromDense(New[float64](WithShape(internal.Min(a.Shape()[0], a.Shape()[1])), WithEngine(e)))
	svd.Values(s.Data())
	if uv {
		svd.UTo(&um)
		svd.VTo(&vm)
		// vm.VFromSVD(&svd)

		u = mk.FromDense(FromMat64[float64](&um, UseUnsafe()))
		v = mk.FromDense(FromMat64[float64](&vm, UseUnsafe()))
	}

	return
}

func (e StdFloat64Engine[T]) Norm(ctx context.Context, t tensor.Basic[float64], ord tensor.NormOrder, axes []int) (retVal tensor.Basic[float64], err error) {
	oneOverOrd := float64(1) / float64(ord)
	ps := func(x float64) float64 { return math.Pow(x, oneOverOrd) }
	norm0 := func(x float64) float64 {
		if x != 0 {
			return 1
		}
		return 0
	}
	normN := func(x float64) float64 { return math.Pow(math.Abs(x), float64(ord)) }

	dims := t.Dims()
	// simple cases
	if len(axes) == 0 {
		if ord.IsUnordered() || (ord.IsFrobenius() && dims == 2) || (ord == tensor.Norm(2) && dims == 1) {
			var ret float64
			ret, err = e.Inner(ctx, t, t)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0")
			}
			ret = math.Sqrt(ret)
			return New[float64](tensor.FromScalar(ret)), nil
		}

		axes = make([]int, dims)
		for i := range axes {
			axes[i] = i
		}
	}
	gd, ok := any(t).(Densor[float64])
	if !ok {
		return nil, errors.Errorf("Cannot get *Dense[float64] out of t")
	}
	d := gd.GetDense()

	return norm[float64, *Dense[float64]](d, ord, axes, math.Sqrt, math.Sqrt, norm0, normN, ps)

}

func (e StdFloat64Engine[T]) Norm2(ctx context.Context, t tensor.Basic[float64]) (float64, error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return math.NaN(), err
	}

	if t.RequiresIterator() {
		return math.NaN(), errors.Errorf(errors.NYIPR, errors.ThisFn(), "tensors requiring iterator")
	}
	retVal := e.blas.Ddot(len(t.Data()), t.Data(), 1, t.Data(), 1)
	return math.Sqrt(retVal), nil
}

func (e StdFloat64Engine[T]) Inner(ctx context.Context, a, b tensor.Basic[float64]) (retVal float64, err error) {
	if err = internal.HandleCtx(ctx); err != nil {
		return 0, err
	}
	A, B := a.Data(), b.Data()
	retVal = e.blas.Ddot(len(A), A, 1, B, 1)
	return
}

func (e StdFloat64Engine[T]) FMA(ctx context.Context, a, x, retVal tensor.Basic[float64]) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}
	var ait, bit, iit Iterator
	var useIter bool
	if ait, bit, iit, useIter, _, err = stdeng.PrepDataVV[float64, float64](a, x, retVal); err != nil {
		return errors.Wrap(err, "FMA")
	}
	if useIter {
		return execution.MulVVIncrIter(a.Data(), x.Data(), retVal.Data(), ait, bit, iit)
	}
	vecf64.IncrMul(a.Data(), x.Data(), retVal.Data())
	return nil
}

func (e StdFloat64Engine[T]) FMAScalar(ctx context.Context, a tensor.Basic[float64], x float64, retVal tensor.Basic[float64]) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}
	panic("NYI")
}

// TODO: FMA Scalar

// MatVecMul is a thin layer over BLAS' DGEMV
// Because DGEMV computes:
//
//	y = αA * x + βy
//
// we set beta to 0, so we don't have to manually zero out the reused/retval tensor data
func (e StdFloat64Engine[T]) MatVecMul(ctx context.Context, a, b, retVal tensor.Basic[float64], incr []float64) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}
	m := a.Shape()[0]
	n := a.Shape()[1]

	tA := blas.NoTrans
	do := a.DataOrder()
	untransposed := !do.IsTransposed()
	if !untransposed {
		m, n = n, m
	}

	var lda int
	switch {
	case do.IsRowMajor() && untransposed:
		lda = n
	case do.IsRowMajor() && !untransposed:
		tA = blas.Trans
		lda = n
	case do.IsColMajor() && untransposed:
		tA = blas.Trans
		lda = m
		m, n = n, m
	case do.IsColMajor() && !untransposed:
		lda = m
		m, n = n, m
	}

	incX, incY := 1, 1 // step size
	var α, β float64 = 1, 0
	e.blas.Dgemv(tA, m, n, α, a.Data(), lda, b.Data(), incX, β, retVal.Data(), incY)
	if len(incr) > 0 {
		vecf64.Add(retVal.Data(), incr)
	}
	return nil
}

func (e StdFloat64Engine[T]) MatMul(ctx context.Context, a, b, retVal tensor.Basic[float64], incr []float64) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}
	m, n, k, lda, ldb, ldc, tA, tB := e.MatMulHelper(a, b, retVal)
	ado := a.DataOrder()
	bdo := b.DataOrder()
	var α, β float64 = 1, 0
	if ado.IsColMajor() && bdo.IsColMajor() {
		e.blas.Dgemm(tA, tB, n, m, k, α, b.Data(), ldb, a.Data(), lda, β, retVal.Data(), ldc)
	} else {
		e.blas.Dgemm(tA, tB, m, n, k, α, a.Data(), lda, b.Data(), ldb, β, retVal.Data(), ldc)
	}
	return nil
}

func (e StdFloat64Engine[T]) Outer(ctx context.Context, a, b, retVal tensor.Basic[float64], incr []float64) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}
	m := a.Shape().TotalSize()
	n := b.Shape().TotalSize()

	incX, incY := 1, 1
	var α float64 = 1
	lda := retVal.Shape()[1]
	if retVal.DataOrder().IsColMajor() {
		// then we simply reshape and call MatMul
		aShape := a.Shape().Clone()
		bShape := b.Shape().Clone()
		if err = a.Reshape(aShape[0], 1); err != nil {
			return err
		}
		if err = b.Reshape(1, bShape[0]); err != nil {
			return err
		}

		if err = e.MatMul(ctx, a, b, retVal, incr); err != nil {
			return err
		}
		if err = b.Reshape(bShape...); err != nil {
			return err
		}
		if err = a.Reshape(aShape...); err != nil {
			return err
		}
		return nil
	}

	e.blas.Dger(m, n, α, a.Data(), incX, b.Data(), incY, retVal.Data(), lda)
	if len(incr) > 0 {
		vecf64.Add(retVal.Data(), incr)
	}
	return nil
}

// NPDot is an implementation of Numpy's Dot.
func (e StdFloat64Engine[T]) NPDot(ctx context.Context, a, b, retVal tensor.Basic[float64], toIncr bool) (scalarRetVal float64, err error) {
	aShp := a.Shape()
	bShp := b.Shape()
	// TODO
	switch {
	case aShp.IsScalar() && bShp.IsScalar():
		return a.Data()[0] * b.Data()[0], nil
	case aShp.IsScalar():
		err = e.MulScalar(ctx, b, a.Data()[0], retVal, true, toIncr)
		return
	case bShp.IsScalar():
		err = e.MulScalar(ctx, a, b.Data()[0], retVal, false, toIncr)
		return
	}

	switch {
	case aShp.IsVector():
		switch {
		case bShp.IsVector():
			scalarRetVal, err = e.Inner(ctx, a, b)
			return
		case bShp.IsMatrix():
			// TODO
		}
	case aShp.IsMatrix():
		switch {
		case bShp.IsVector():
			var incr []float64
			if toIncr {
				incr = retVal.Data()
			}
			err = e.MatVecMul(ctx, a, b, retVal, incr)
			return
		case bShp.IsMatrix():
			var incr []float64
			if toIncr {
				incr = retVal.Data()
			}
			err = e.MatMul(ctx, a, b, retVal, incr)
			return
		}
	}

	// TODO: TensorMul

	panic("NYI")
}
