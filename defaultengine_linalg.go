package tensor

import (
	"context"
	"reflect"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/dtype"
)

// Trace returns the trace of a matrix (i.e. the sum of the diagonal elements). If the Tensor provided is not a matrix, it will return an error
func (e StdEng) Trace(ctx context.Context, t Tensor) (retVal interface{}, err error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	if t.Dims() != 2 {
		err = errors.Errorf(dimMismatch, 2, t.Dims())
		return
	}

	if err = dtype.TypeClassCheck(t.Dtype(), dtype.Number); err != nil {
		return nil, errors.Wrap(err, "Trace")
	}

	rstride := t.Strides()[0]
	cstride := t.Strides()[1]

	r := t.Shape()[0]
	c := t.Shape()[1]

	m := MinInt(r, c)
	stride := rstride + cstride

	switch data := t.Data().(type) {
	case []int:
		var trace int
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int8:
		var trace int8
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int16:
		var trace int16
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int32:
		var trace int32
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int64:
		var trace int64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint:
		var trace uint
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint8:
		var trace uint8
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint16:
		var trace uint16
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint32:
		var trace uint32
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint64:
		var trace uint64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []float32:
		var trace float32
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []float64:
		var trace float64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []complex64:
		var trace complex64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []complex128:
		var trace complex128
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	}
	return
}

func (e StdEng) Dot(x, y Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	fo := ParseFuncOpts(opts...)
	ctx := fo.Context()
	if err = handleCtx(ctx); err != nil {
		return nil, err
	}

	if _, ok := x.(DenseTensor); !ok {
		err = errors.Errorf("Engine only supports working on x that is a DenseTensor. Got %T instead", x)
		return
	}

	if _, ok := y.(DenseTensor); !ok {
		err = errors.Errorf("Engine only supports working on y that is a DenseTensor. Got %T instead", y)
		return
	}

	var a, b DenseTensor
	if a, err = getFloatDenseTensor(x); err != nil {
		err = errors.Wrapf(err, opFail, "Dot")
		return
	}
	if b, err = getFloatDenseTensor(y); err != nil {
		err = errors.Wrapf(err, opFail, "Dot")
		return
	}

	var reuse, incr DenseTensor
	if reuse, err = getFloatDenseTensor(fo.reuse); err != nil {
		err = errors.Wrapf(err, opFail, "Dot - reuse")
		return

	}

	if incr, err = getFloatDenseTensor(fo.incr); err != nil {
		err = errors.Wrapf(err, opFail, "Dot - incr")
		return
	}

	switch {
	case a.IsScalar() && b.IsScalar():
		var res interface{}
		switch a.Dtype().Kind() {
		case reflect.Float64:
			res = a.GetF64(0) * b.GetF64(0)
		case reflect.Float32:
			res = a.GetF32(0) * b.GetF32(0)
		}

		switch {
		case incr != nil:
			if !incr.IsScalar() {
				err = errors.Errorf(shapeMismatch, ScalarShape(), incr.Shape())
				return
			}
			if err = e.E.MulIncr(a.Dtype().Type, a.hdr(), b.hdr(), incr.hdr()); err != nil {
				err = errors.Wrapf(err, opFail, "Dot scalar incr")
				return

			}
			retVal = incr
		case reuse != nil:
			reuse.Set(0, res)
			reuse.reshape()
			retVal = reuse
		default:
			retVal = New(FromScalar(res))
		}
		return
	case a.IsScalar():
		switch {
		case incr != nil:
			return Mul(a.ScalarValue(), b, WithIncr(incr))
		case reuse != nil:
			return Mul(a.ScalarValue(), b, WithReuse(reuse))
		}
		// default moved out
		return Mul(a.ScalarValue(), b)
	case b.IsScalar():
		switch {
		case incr != nil:
			return Mul(a, b.ScalarValue(), WithIncr(incr))
		case reuse != nil:
			return Mul(a, b.ScalarValue(), WithReuse(reuse))
		}
		return Mul(a, b.ScalarValue())
	}

	switch {
	case a.IsVector():
		switch {
		case b.IsVector():
			// check size
			if a.len() != b.len() {
				err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
				return
			}
			var ret interface{}
			if ret, err = e.Inner(ctx, a, b); err != nil {
				return nil, errors.Wrapf(err, opFail, "Dot")
			}
			return New(FromScalar(ret)), nil
		case b.IsMatrix():
			b.T()
			defer b.UT()
			switch {
			case reuse != nil && incr != nil:
				return b.MatVecMul(a, WithReuse(reuse), WithIncr(incr))
			case reuse != nil:
				return b.MatVecMul(a, WithReuse(reuse))
			case incr != nil:
				return b.MatVecMul(a, WithIncr(incr))
			default:
			}
			return b.MatVecMul(a)
		default:

		}
	case a.IsMatrix():
		switch {
		case b.IsVector():
			switch {
			case reuse != nil && incr != nil:
				return a.MatVecMul(b, WithReuse(reuse), WithIncr(incr))
			case reuse != nil:
				return a.MatVecMul(b, WithReuse(reuse))
			case incr != nil:
				return a.MatVecMul(b, WithIncr(incr))
			default:
			}
			return a.MatVecMul(b)

		case b.IsMatrix():
			switch {
			case reuse != nil && incr != nil:
				return a.MatMul(b, WithReuse(reuse), WithIncr(incr))
			case reuse != nil:
				return a.MatMul(b, WithReuse(reuse))
			case incr != nil:
				return a.MatMul(b, WithIncr(incr))
			default:
			}
			return a.MatMul(b)
		default:
		}
	default:
	}

	as := a.Shape()
	bs := b.Shape()
	axesA := BorrowInts(1)
	axesB := BorrowInts(1)
	defer ReturnInts(axesA)
	defer ReturnInts(axesB)

	var lastA, secondLastB int

	lastA = len(as) - 1
	axesA[0] = lastA
	if len(bs) >= 2 {
		secondLastB = len(bs) - 2
	} else {
		secondLastB = 0
	}
	axesB[0] = secondLastB

	if as[lastA] != bs[secondLastB] {
		err = errors.Errorf(shapeMismatch, as, bs)
		return
	}

	var rd *Dense
	if rd, err = a.TensorMul(b, axesA, axesB); err != nil {
		panic(err)
	}

	if reuse != nil {
		copyDense(reuse, rd)
		ap := rd.Info().Clone()
		reuse.setAP(&ap)
		defer ReturnTensor(rd)
		// swap out the underlying data and metadata
		// reuse.data, rd.data = rd.data, reuse.data
		// reuse.AP, rd.AP = rd.AP, reuse.AP
		// defer ReturnTensor(rd)

		retVal = reuse
	} else {
		retVal = rd
	}

	return
}

// TODO: make it take DenseTensor
func (e StdEng) SVD(ctx context.Context, a Tensor, uv, full bool) (s, u, v Tensor, err error) {
	if err = handleCtx(ctx); err != nil {
		return nil, nil, nil, err
	}

	var t *Dense
	var ok bool
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, nil, errors.Wrapf(err, "opFail %v", "SVD")
	}
	if t, ok = a.(*Dense); !ok {
		return nil, nil, nil, errors.Errorf("StdEng only performs SVDs for DenseTensors. Got %T instead", a)
	}
	if err = dtype.TypeClassCheck(a.Dtype(), dtype.Floats); err != nil {
		return nil, nil, nil, errors.Errorf("StdEng can only perform SVDs for float64 and float32 type. Got tensor of %v instead", t.Dtype())
	}

	if !t.IsMatrix() {
		return nil, nil, nil, errors.Errorf(dimMismatch, 2, t.Dims())
	}

	var m *mat.Dense
	var svd mat.SVD

	if m, err = ToMat64(t, UseUnsafe()); err != nil {
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
	var um, vm mat.Dense
	s = recycledDense(Float64, Shape{MinInt(t.Shape()[0], t.Shape()[1])}, WithEngine(e))
	svd.Values(s.Data().([]float64))
	if uv {
		svd.UTo(&um)
		svd.VTo(&vm)
		// vm.VFromSVD(&svd)

		u = FromMat64(&um, UseUnsafe(), As(t.t))
		v = FromMat64(&vm, UseUnsafe(), As(t.t))
	}

	return
}

// Inner is a thin layer over BLAS's D/Sdot.
// It returns a scalar value, wrapped in an interface{}, which is not quite nice.
func (e StdEng) Inner(ctx context.Context, a, b Tensor) (retVal interface{}, err error) {
	if err = handleCtx(ctx); err != nil {
		return nil, err // this err will be noopError{}, no need to wrap.
	}

	var ad, bd DenseTensor
	if ad, bd, err = e.checkTwoFloatComplexTensors(a, b); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Inner")
	}

	switch A := ad.Data().(type) {
	case []float32:
		B := bd.Float32s()
		retVal = whichblas.Sdot(len(A), A, 1, B, 1)
	case []float64:
		B := bd.Float64s()
		retVal = whichblas.Ddot(len(A), A, 1, B, 1)
	case []complex64:
		B := bd.Complex64s()
		retVal = whichblas.Cdotu(len(A), A, 1, B, 1)
	case []complex128:
		B := bd.Complex128s()
		retVal = whichblas.Zdotu(len(A), A, 1, B, 1)
	}
	return
}

// MatVecMul is a thin layer over BLAS' DGEMV
// Because DGEMV computes:
// 		y = αA * x + βy
// we set beta to 0, so we don't have to manually zero out the reused/retval tensor data
func (e StdEng) MatVecMul(ctx context.Context, a, b, prealloc Tensor) (err error) {
	if err := handleCtx(ctx); err != nil {
		return err
	}

	// check all are DenseTensors
	var ad, bd, pd DenseTensor
	if ad, bd, pd, err = e.checkThreeFloatComplexTensors(a, b, prealloc); err != nil {
		return errors.Wrapf(err, opFail, "StdEng.MatVecMul")
	}

	m := ad.oshape()[0]
	n := ad.oshape()[1]

	tA := blas.NoTrans
	do := a.DataOrder()
	z := ad.oldAP().IsZero()

	var lda int
	switch {
	case do.IsRowMajor() && z:
		lda = n
	case do.IsRowMajor() && !z:
		tA = blas.Trans
		lda = n
	case do.IsColMajor() && z:
		tA = blas.Trans
		lda = m
		m, n = n, m
	case do.IsColMajor() && !z:
		lda = m
		m, n = n, m
	}

	incX, incY := 1, 1 // step size

	// ASPIRATIONAL TODO: different incX and incY
	// TECHNICAL DEBT. TECHDEBT. TECH DEBT
	// Example use case:
	// log.Printf("a %v %v", ad.Strides(), ad.ostrides())
	// log.Printf("b %v", b.Strides())
	// incX := a.Strides()[0]
	// incY = b.Strides()[0]

	switch A := ad.Data().(type) {
	case []float64:
		x := bd.Float64s()
		y := pd.Float64s()
		alpha, beta := float64(1), float64(0)
		whichblas.Dgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case []float32:
		x := bd.Float32s()
		y := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		whichblas.Sgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case []complex64:
		x := bd.Complex64s()
		y := pd.Complex64s()
		var alpha, beta complex64 = complex(1, 0), complex(0, 0)
		whichblas.Cgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case []complex128:
		x := bd.Complex128s()
		y := pd.Complex128s()
		var alpha, beta complex128 = complex(1, 0), complex(0, 0)
		whichblas.Zgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	default:
		return nyierr(typeNYI, bd.Data())
	}

	return nil
}

// MatMul is a thin layer over DGEMM.
// DGEMM computes:
//		C = αA * B +  βC
// To prevent needless zeroing out of the slice, we just set β to 0
func (e StdEng) MatMul(ctx context.Context, a, b, prealloc Tensor) (err error) {
	if err := handleCtx(ctx); err != nil {
		return err
	}

	// check all are DenseTensors
	var ad, bd, pd DenseTensor
	if ad, bd, pd, err = e.checkThreeFloatComplexTensors(a, b, prealloc); err != nil {
		return errors.Wrapf(err, opFail, "StdEng.MatMul")
	}

	ado := a.DataOrder()
	bdo := b.DataOrder()
	cdo := prealloc.DataOrder()

	// get result shapes. k is the shared dimension
	// a is (m, k)
	// b is (k, n)
	// c is (m, n)
	var m, n, k int
	m = ad.Shape()[0]
	k = ad.Shape()[1]
	n = bd.Shape()[1]

	// wrt the strides, we use the original strides, because that's what BLAS needs, instead of calling .Strides()
	// lda in colmajor = number of rows;
	// lda in row major = number of cols
	var lda, ldb, ldc int
	switch {
	case ado.IsColMajor():
		lda = m
	case ado.IsRowMajor():
		lda = k
	}

	switch {
	case bdo.IsColMajor():
		ldb = bd.Shape()[0]
	case bdo.IsRowMajor():
		ldb = n
	}

	switch {
	case cdo.IsColMajor():
		ldc = prealloc.Shape()[0]
	case cdo.IsRowMajor():
		ldc = prealloc.Shape()[1]
	}

	// check for trans
	tA, tB := blas.NoTrans, blas.NoTrans
	if !ad.oldAP().IsZero() {
		tA = blas.Trans
		if ado.IsRowMajor() {
			lda = m
		} else {
			lda = k
		}
	}
	if !bd.oldAP().IsZero() {
		tB = blas.Trans
		if bdo.IsRowMajor() {
			ldb = bd.Shape()[0]
		} else {
			ldb = bd.Shape()[1]
		}
	}

	switch A := ad.Data().(type) {
	case []float64:
		B := bd.Float64s()
		C := pd.Float64s()
		alpha, beta := float64(1), float64(0)
		if ado.IsColMajor() && bdo.IsColMajor() {
			whichblas.Dgemm(tA, tB, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
		} else {
			whichblas.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
		}
	case []float32:
		B := bd.Float32s()
		C := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		if ado.IsColMajor() && bdo.IsColMajor() {
			whichblas.Sgemm(tA, tB, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
		} else {
			whichblas.Sgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
		}
	case []complex64:
		B := bd.Complex64s()
		C := pd.Complex64s()
		var alpha, beta complex64 = complex(1, 0), complex(0, 0)
		if ado.IsColMajor() && bdo.IsColMajor() {
			whichblas.Cgemm(tA, tB, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
		} else {
			whichblas.Cgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
		}
	case []complex128:
		B := bd.Complex128s()
		C := pd.Complex128s()
		var alpha, beta complex128 = complex(1, 0), complex(0, 0)
		if ado.IsColMajor() && bdo.IsColMajor() {
			whichblas.Zgemm(tA, tB, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
		} else {
			whichblas.Zgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
		}
	default:
		return nyierr(typeNYI, ad.Data())

	}
	return
}

// Outer is a thin wrapper over S/Dger
func (e StdEng) Outer(ctx context.Context, a, b, prealloc Tensor) (err error) {
	if err = handleCtx(ctx); err != nil {
		return err
	}

	// check all are DenseTensors
	var ad, bd, pd DenseTensor
	if ad, bd, pd, err = e.checkThreeFloatComplexTensors(a, b, prealloc); err != nil {
		return errors.Wrapf(err, opFail, "StdEng.Outer")
	}

	m := ad.Size()
	n := bd.Size()
	pdo := pd.DataOrder()

	// the stride of a Vector is always going to be [1],
	// incX := t.Strides()[0]
	// incY := other.Strides()[0]
	incX, incY := 1, 1
	// lda := pd.Strides()[0]
	var lda int
	switch {
	case pdo.IsColMajor():
		aShape := a.Shape().Clone()
		bShape := b.Shape().Clone()
		if err = a.Reshape(aShape[0], 1); err != nil {
			return err
		}
		if err = b.Reshape(1, bShape[0]); err != nil {
			return err
		}

		if err = e.MatMul(ctx, a, b, prealloc); err != nil {
			return err
		}

		if err = b.Reshape(bShape...); err != nil {
			return
		}
		if err = a.Reshape(aShape...); err != nil {
			return
		}
		return nil

	case pdo.IsRowMajor():
		lda = pd.Shape()[1]
	}

	switch x := ad.Data().(type) {
	case []float64:
		y := bd.Float64s()
		A := pd.Float64s()
		alpha := float64(1)
		whichblas.Dger(m, n, alpha, x, incX, y, incY, A, lda)
	case []float32:
		y := bd.Float32s()
		A := pd.Float32s()
		alpha := float32(1)
		whichblas.Sger(m, n, alpha, x, incX, y, incY, A, lda)
	case []complex64:
		y := bd.Complex64s()
		A := pd.Complex64s()
		var alpha complex64 = complex(1, 0)
		whichblas.Cgeru(m, n, alpha, x, incX, y, incY, A, lda)
	case []complex128:
		y := bd.Complex128s()
		A := pd.Complex128s()
		var alpha complex128 = complex(1, 0)
		whichblas.Zgeru(m, n, alpha, x, incX, y, incY, A, lda)
	default:
		return nyierr(typeNYI, b.Data())
	}
	return nil
}

/* UNEXPORTED UTILITY FUNCTIONS */

func (e StdEng) checkTwoFloatTensors(a, b Tensor) (ad, bd DenseTensor, err error) {
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}
	if err = e.checkAccessible(b); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}

	if a.Dtype() != b.Dtype() {
		return nil, nil, errors.New("Expected a and b to have the same Dtype")
	}

	if ad, err = getFloatDenseTensor(a); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors expects a to be be a DenseTensor")
	}
	if bd, err = getFloatDenseTensor(b); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors expects b to be be a DenseTensor")
	}
	return
}

func (e StdEng) checkThreeFloatTensors(a, b, ret Tensor) (ad, bd, retVal DenseTensor, err error) {
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkThreeTensors: a is not accessible")
	}
	if err = e.checkAccessible(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkThreeTensors: a is not accessible")
	}
	if err = e.checkAccessible(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkThreeTensors: ret is not accessible")
	}

	if a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}

	if ad, err = getFloatDenseTensor(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects a to be be a DenseTensor")
	}
	if bd, err = getFloatDenseTensor(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects b to be be a DenseTensor")
	}
	if retVal, err = getFloatDenseTensor(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects retVal to be be a DenseTensor")
	}
	return
}

func (e StdEng) checkTwoFloatComplexTensors(a, b Tensor) (ad, bd DenseTensor, err error) {
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}
	if err = e.checkAccessible(b); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}

	if a.Dtype() != b.Dtype() {
		return nil, nil, errors.New("Expected a and b to have the same Dtype")
	}

	if ad, err = getFloatComplexDenseTensor(a); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors expects a to be be a DenseTensor")
	}
	if bd, err = getFloatComplexDenseTensor(b); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors expects b to be be a DenseTensor")
	}
	return
}

func (e StdEng) checkThreeFloatComplexTensors(a, b, ret Tensor) (ad, bd, retVal DenseTensor, err error) {
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkThreeTensors: a is not accessible")
	}
	if err = e.checkAccessible(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkThreeTensors: a is not accessible")
	}
	if err = e.checkAccessible(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkThreeTensors: ret is not accessible")
	}

	if a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}

	if ad, err = getFloatComplexDenseTensor(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects a to be be a DenseTensor")
	}
	if bd, err = getFloatComplexDenseTensor(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects b to be be a DenseTensor")
	}
	if retVal, err = getFloatComplexDenseTensor(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects retVal to be be a DenseTensor")
	}
	return
}
