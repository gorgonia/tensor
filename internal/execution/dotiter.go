package execution

import (
	"runtime"
	"sync"

	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gonum.org/v1/gonum/blas"
)

const (
	badTranspose = "Bad Transpose for %v: %v"
	mLT0         = "m: %d < 0"
	nLT0         = "n: %d < 0"
	kLT0         = "k: %d < 0"
	badLd        = "Bad Leading Dimension %s. %d < max(1, %d). Matrix is %s. Leading dimension should be < %v (%d)"
	shortA       = "Short A"
	shortB       = "Short B"
	shortC       = "Short C"

	blockSize    = 64
	minParBlocks = 4
)

// DoIterA is a function that performs a dot-iteration taking two functions:
//   - `op`  - the elementwise operation
//   - `red` - an associative reduction function
//
// An example would be inner product:
//   - `op` : func(a, b int) int { return a * b }
//   - `red`: func(a, b int) int { return a + b }
//
// Both `a` and `b` has to be of length > 0. Both `a` and `b` must have the same
// length
func DotIterA[DT any](red, op func(DT, DT) DT, a, b []DT) (retVal DT) {
	if len(a) == 0 {
		panic("Expected `a` and `b` to have at least one element")
	}
	retVal = op(a[0], b[0])
	if len(a) < 2 {
		return
	}

	// prevents bounds checks
	a = a[1:len(a)]
	b = b[1:len(a)]

	// The actual loop
	for i, v := range a {
		retVal = red(retVal, op(v, b[i]))
	}
	return
}

// Afxry generalizes Apxy
func Afxry[DT any](red, op func(DT, DT) DT, a DT, x, y []DT) {
	x = x[:len(x)]
	y = y[:len(x)]
	for i, v := range x {
		y[i] = red(y[i], op(a, v))
	}
}

func AfxryInc[DT any](red, op func(DT, DT) DT, a DT, x, y []DT, n, incX, incY, ix, iy int) {
	for i := 0; i < n; i++ {
		y[iy] = red(y[iy], op(a, x[ix]))
		ix += incX
		iy += incY
	}
}

// GeDOR generalizes GeMM
func GeDOR[DT any](red, op func(DT, DT) DT, tA, tB blas.Transpose, m, n, k int, a []DT, lda int, b []DT, ldb int, c []DT, ldc int) (err error) {
	switch tA {
	default:
		return errors.Errorf(badTranspose, "A", tA)
	case blas.NoTrans, blas.Trans, blas.ConjTrans:
	}
	switch tB {
	default:
		return errors.Errorf(badTranspose, "B", tB)
	case blas.NoTrans, blas.Trans, blas.ConjTrans:
	}
	if m < 0 {
		return errors.Errorf(mLT0, m)
	}
	if n < 0 {
		return errors.Errorf(nLT0, n)
	}
	if k < 0 {
		return errors.Errorf(kLT0, k)
	}
	aTrans := tA == blas.Trans || tA == blas.ConjTrans
	if aTrans {
		if lda < internal.Max(1, m) {
			return errors.Errorf(badLd, "A", lda, m, "transposed", "m", m)
		}
	} else {
		if lda < internal.Max(1, k) {
			return errors.Errorf(badLd, "A", lda, k, "not transposed", "k", k)
		}
	}
	bTrans := tB == blas.Trans || tB == blas.ConjTrans
	if bTrans {
		if ldb < internal.Max(1, k) {
			return errors.Errorf(badLd, "B", ldb, k, "transposed", "k", k)
		}
	} else {
		if ldb < internal.Max(1, n) {
			return errors.Errorf(badLd, "B", ldb, n, "transposed", "n", n)
		}
	}
	if ldc < internal.Max(1, n) {
		return errors.Errorf(badLd, "C", ldc, n, "", "n", n)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if aTrans {
		if len(a) < (k-1)*lda+m {
			return errors.New(shortA)
		}
	} else {
		if len(a) < (m-1)*lda+k {
			return errors.New(shortA)
		}
	}
	if bTrans {
		if len(b) < (n-1)*ldb+k {
			return errors.New(shortB)
		}
	} else {
		if len(b) < (k-1)*ldb+n {
			return errors.New(shortB)
		}
	}
	if len(c) < (m-1)*ldc+n {
		return errors.New(shortC)
	}

	gedorParallel(red, op, aTrans, bTrans, m, n, k, a, lda, b, ldb, c, ldc)
	return nil
}

func gedorParallel[DT any](red, op func(DT, DT) DT, aTrans, bTrans bool, m, n, k int, a []DT, lda int, b []DT, ldb int, c []DT, ldc int) {
	// dgemmParallel computes a parallel matrix multiplication by partitioning
	// a and b into sub-blocks, and updating c with the multiplication of the sub-block
	// In all cases,
	// A = [ 	A_11	A_12 ... 	A_1j
	//		A_21	A_22 ...	A_2j
	//			...
	//		A_i1	A_i2 ...	A_ij]
	//
	// and same for B. All of the submatrix sizes are blockSize×blockSize except
	// at the edges.
	//
	// In all cases, there is one dimension for each matrix along which
	// C must be updated sequentially.
	// Cij = \sum_k Aik Bki,	(A * B)
	// Cij = \sum_k Aki Bkj,	(Aᵀ * B)
	// Cij = \sum_k Aik Bjk,	(A * Bᵀ)
	// Cij = \sum_k Aki Bjk,	(Aᵀ * Bᵀ)
	//
	// This code computes one {i, j} block sequentially along the k dimension,
	// and computes all of the {i, j} blocks concurrently. This
	// partitioning allows Cij to be updated in-place without race-conditions.
	// Instead of launching a goroutine for each possible concurrent computation,
	// a number of worker goroutines are created and channels are used to pass
	// available and completed cases.
	//
	// http://alexkr.com/docs/matrixmult.pdf is a good reference on matrix-matrix
	// multiplies, though this code does not copy matrices to attempt to eliminate
	// cache misses.

	maxKLen := k
	parBlocks := blocks(m, blockSize) * blocks(n, blockSize)
	if parBlocks < minParBlocks {
		// The matrix multiplication is small in the dimensions where it can be
		// computed concurrently. Just do it in serial.
		gedorSerial(red, op, aTrans, bTrans, m, n, k, a, lda, b, ldb, c, ldc)
		return
	}

	// workerLimit acts a number of maximum concurrent workers,
	// with the limit set to the number of procs available.
	workerLimit := make(chan struct{}, runtime.GOMAXPROCS(0))

	// wg is used to wait for all
	var wg sync.WaitGroup
	wg.Add(parBlocks)
	defer wg.Wait()

	for i := 0; i < m; i += blockSize {
		for j := 0; j < n; j += blockSize {
			workerLimit <- struct{}{}
			go func(i, j int) {
				defer func() {
					wg.Done()
					<-workerLimit
				}()

				leni := blockSize
				if i+leni > m {
					leni = m - i
				}
				lenj := blockSize
				if j+lenj > n {
					lenj = n - j
				}

				cSub := sliceView(c, ldc, i, j, leni, lenj)

				// Compute A_ik B_kj for all k
				for k := 0; k < maxKLen; k += blockSize {
					lenk := blockSize
					if k+lenk > maxKLen {
						lenk = maxKLen - k
					}
					var aSub, bSub []DT
					if aTrans {
						aSub = sliceView(a, lda, k, i, lenk, leni)
					} else {
						aSub = sliceView(a, lda, i, k, leni, lenk)
					}
					if bTrans {
						bSub = sliceView(b, ldb, j, k, lenj, lenk)
					} else {
						bSub = sliceView(b, ldb, k, j, lenk, lenj)
					}
					gedorSerial(red, op, aTrans, bTrans, leni, lenj, lenk, aSub, lda, bSub, ldb, cSub, ldc)
				}
			}(i, j)
		}
	}
}

// dgemmSerial is serial matrix multiply
func gedorSerial[DT any](red, op func(DT, DT) DT, aTrans, bTrans bool, m, n, k int, a []DT, lda int, b []DT, ldb int, c []DT, ldc int) {
	switch {
	case !aTrans && !bTrans:
		gedorSerialNotNot(red, op, m, n, k, a, lda, b, ldb, c, ldc)
		return
	case aTrans && !bTrans:
		gedorSerialTransNot(red, op, m, n, k, a, lda, b, ldb, c, ldc)
		return
	case !aTrans && bTrans:
		gedorSerialNotTrans(red, op, m, n, k, a, lda, b, ldb, c, ldc)
		return
	case aTrans && bTrans:
		gedorSerialTransTrans(red, op, m, n, k, a, lda, b, ldb, c, ldc)
		return
	default:
		panic("unreachable")
	}
}

// gedorSerial where neither a nor b are transposed
func gedorSerialNotNot[DT any](red, op func(DT, DT) DT, m, n, k int, a []DT, lda int, b []DT, ldb int, c []DT, ldc int) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for i := 0; i < m; i++ {
		ctmp := c[i*ldc : i*ldc+n]
		for l, v := range a[i*lda : i*lda+k] {
			Afxry(red, op, v, b[l*ldb:l*ldb+n], ctmp)
		}
	}
}

// gedorSerial where neither a is transposed and b is not
func gedorSerialTransNot[DT any](red, op func(DT, DT) DT, m, n, k int, a []DT, lda int, b []DT, ldb int, c []DT, ldc int) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for l := 0; l < k; l++ {
		btmp := b[l*ldb : l*ldb+n]
		for i, v := range a[l*lda : l*lda+m] {
			ctmp := c[i*ldc : i*ldc+n]
			Afxry(red, op, v, btmp, ctmp)
		}
	}
}

// gedorSerial where neither a is not transposed and b is
func gedorSerialNotTrans[DT any](red, op func(DT, DT) DT, m, n, k int, a []DT, lda int, b []DT, ldb int, c []DT, ldc int) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for i := 0; i < m; i++ {
		atmp := a[i*lda : i*lda+k]
		ctmp := c[i*ldc : i*ldc+n]
		for j := 0; j < n; j++ {
			ctmp[j] = red(ctmp[j], DotIterA(red, op, atmp, b[j*ldb:j*ldb+k]))
		}
	}
}

// gedorSerial where both are transposed
func gedorSerialTransTrans[DT any](red, op func(DT, DT) DT, m, n, k int, a []DT, lda int, b []DT, ldb int, c []DT, ldc int) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for l := 0; l < k; l++ {
		for i, v := range a[l*lda : l*lda+m] {
			ctmp := c[i*ldc : i*ldc+n]
			AfxryInc(red, op, v, b[l:], ctmp, n, ldb, 1, 0, 0)
		}
	}
}

func sliceView[DT any](a []DT, lda, i, j, r, c int) []DT { return a[i*lda+j : (i+r-1)*lda+j+c] }

// blocks returns the number of divisions of the dimension length with the given
// block size.
func blocks(dim, bsize int) int { return (dim + bsize - 1) / bsize }
