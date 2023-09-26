package stdeng

import (
	"gonum.org/v1/gonum/blas"
	"gorgonia.org/shapes"
)

func convertToBool[DT comparable](data []DT) []bool {
	retVal := make([]bool, len(data), cap(data))
	var z DT
	for i := range data {
		retVal[i] = data[i] != z
	}
	return retVal
}

// MatMulHelper is a helper function to extract all the necessary variables for a BLAS call to MatMul (or similar).
// Notation used is in accordance to BLAS's notation:
//
//	a has shape (m, k)
//	b has shape (k, n)
//	c has shape (m, n)
func (e StdEng[DT, T]) MatMulHelper(a, b, retVal T, asShapes ...shapes.Shape) (m, n, k, lda, ldb, ldc int, tA, tB blas.Transpose) {
	ado := a.DataOrder()
	bdo := b.DataOrder()
	cdo := retVal.DataOrder()

	aShape, bShape, cShape := a.Shape(), b.Shape(), retVal.Shape()
	if len(asShapes) == 3 {
		aShape = asShapes[0]
		bShape = asShapes[1]
		cShape = asShapes[2]
	}

	// get result shapes. k is the shared dimension
	// a is (m, k)
	// b is (k, n)
	// c is (m, n)
	m = aShape[0]
	k = aShape[1]
	n = bShape[1]

	// wrt the strides, we use the original strides, because that's what BLAS needs, instead of calling .Strides()
	// lda in colmajor = number of rows;
	// lda in row major = number of cols
	switch {
	case ado.IsColMajor():
		lda = m
	case ado.IsRowMajor():
		lda = k
	}

	switch {
	case bdo.IsColMajor():
		ldb = k
	case bdo.IsRowMajor():
		ldb = n
	}

	switch {
	case cdo.IsColMajor():
		ldc = cShape[0]
	case cdo.IsRowMajor():
		ldc = cShape[1]
	}
	// check for trans
	tA, tB = blas.NoTrans, blas.NoTrans
	if ado.IsTransposed() {
		tA = blas.Trans
		if ado.IsRowMajor() {
			lda = m
		} else {
			lda = k
		}
	}
	if bdo.IsTransposed() {
		tB = blas.Trans
		if bdo.IsRowMajor() {
			ldb = bShape[0]
		} else {
			ldb = bShape[1]
		}
	}
	return
}
