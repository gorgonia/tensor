package dense

import (
	"context"

	"github.com/chewxy/math32"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/tensor"
	stdeng "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/execution"
	"gorgonia.org/vecf32"
)

type StdFloat32Engine[T tensor.Tensor[float32, T]] struct {
	stdeng.StdOrderedNumEngine[float32, T]
	blas gonum.Implementation // the default BLAS is gonum's BLAS
}

func (e StdFloat32Engine[T]) BasicEng() Engine {
	return stdeng.StdOrderedNumEngine[float32, tensor.Basic[float32]]{}
}
func (e StdFloat32Engine[T]) Workhorse() Engine { return e }

func (e StdFloat32Engine[T]) SVD(ctx context.Context, a T, uv, full bool) (s, u, v T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return s, u, v, err
	}

	if !tensor.IsMatrix(a) {
		return s, u, v, errors.Errorf(errors.DimMismatch, 2, a.Dims())
	}

	var m *mat.Dense
	var svd mat.SVD

	if m, err = ToMat64[float32](a, UseUnsafe()); err != nil {
		return
	}

	var ok bool
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
	mk, ok := any(z).(FromDenser[float32, T])
	if !ok {
		return s, u, v, errors.Errorf("Unable to build a constructor from %T", z)
	}

	var um, vm mat.Dense

	shp := internal.Min(a.Shape()[0], a.Shape()[1])
	s64 := make([]float64, shp)
	svd.Values(s64)
	s = mk.FromDense(New[float32](WithShape(shp), WithBacking(convert[float32](s64))))
	if uv {
		svd.UTo(&um)
		svd.VTo(&vm)

		u = mk.FromDense(FromMat64[float32](&um, UseUnsafe()))
		v = mk.FromDense(FromMat64[float32](&vm, UseUnsafe()))
	}

	return
}

func (e StdFloat32Engine[T]) Norm(ctx context.Context, t T, ord tensor.NormOrder, axes []int) (retVal tensor.Basic[float32], err error) {
	oneOverOrd := float32(1) / float32(ord)
	ps := func(x float32) float32 { return math32.Pow(x, oneOverOrd) }
	norm0 := func(x float32) float32 {
		if x != 0 {
			return 1
		}
		return 0
	}
	normN := func(x float32) float32 { return math32.Pow(math32.Abs(x), float32(ord)) }

	dims := t.Dims()
	// simple cases
	if len(axes) == 0 {
		if ord.IsUnordered() || (ord.IsFrobenius() && dims == 2) || (ord == tensor.Norm(2) && dims == 1) {
			var ret float32
			ret, err = e.Inner(ctx, t, t)
			if err != nil {
				return retVal, errors.Wrapf(err, errors.OpFail, "Norm-0")
			}
			ret = math32.Sqrt(ret)
			return New[float32](tensor.FromScalar(ret)), nil
		}

		axes = make([]int, dims)
		for i := range axes {
			axes[i] = i
		}
	}

	return norm[float32, T](t, ord, axes, math32.Sqrt, math32.Sqrt, norm0, normN, ps)
}

func (e StdFloat32Engine[T]) Norm2(ctx context.Context, t T) (float32, error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return math32.NaN(), err
	}

	if t.RequiresIterator() {
		return math32.NaN(), errors.Errorf(errors.NYIPR, errors.ThisFn(), "tensors requiring iterator")
	}
	retVal := e.blas.Sdot(len(t.Data()), t.Data(), 1, t.Data(), 1)
	return math32.Sqrt(retVal), nil
}

func (e StdFloat32Engine[T]) Inner(ctx context.Context, a, b T) (retVal float32, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return 0, err
	}

	A, B := a.Data(), b.Data()
	retVal = e.blas.Sdot(len(A), A, 1, B, 1)
	return
}

func (e StdFloat32Engine[T]) FMA(ctx context.Context, a, x, retVal T) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}

	var ait, bit, iit Iterator
	var useIter bool
	if ait, bit, iit, useIter, _, err = stdeng.PrepDataVV[float32, float32](a, x, retVal); err != nil {
		return errors.Wrap(err, "FMA")
	}
	if useIter {
		return execution.MulVVIncrIter(a.Data(), x.Data(), retVal.Data(), ait, bit, iit)
	}
	vecf32.IncrMul(a.Data(), x.Data(), retVal.Data())
	return nil
}

func (e StdFloat32Engine[T]) MatVecMul(ctx context.Context, a, b, retVal T, incr []float32) (err error) {
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
	var α, β float32 = 1, 0
	e.blas.Sgemv(tA, m, n, α, a.Data(), lda, b.Data(), incX, β, retVal.Data(), incY)
	if len(incr) > 0 {
		vecf32.Add(retVal.Data(), incr)
	}
	return nil
}

func (e StdFloat32Engine[T]) MatMul(ctx context.Context, a, b, retVal T, incr []float32) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}

	m, n, k, lda, ldb, ldc, tA, tB := e.MatMulHelper(a, b, retVal)
	ado := a.DataOrder()
	bdo := b.DataOrder()

	var α, β float32 = 1, 0
	if ado.IsColMajor() && bdo.IsColMajor() {
		e.blas.Sgemm(tA, tB, n, m, k, α, b.Data(), ldb, a.Data(), lda, β, retVal.Data(), ldc)
	} else {
		e.blas.Sgemm(tA, tB, m, n, k, α, a.Data(), lda, b.Data(), ldb, β, retVal.Data(), ldc)
	}

	if len(incr) > 0 {
		vecf32.Add(retVal.Data(), incr)
	}
	return nil
}

func (e StdFloat32Engine[T]) Outer(ctx context.Context, a, b, retVal T, incr []float32) (err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return err
	}
	m := a.Shape().TotalSize()
	n := b.Shape().TotalSize()

	incX, incY := 1, 1
	var α float32 = 1
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

	e.blas.Sger(m, n, α, a.Data(), incX, b.Data(), incY, retVal.Data(), lda)
	if len(incr) > 0 {
		vecf32.Add(retVal.Data(), incr)
	}
	return nil
}
