package execution

import (
	"testing"

	gutils "gorgonia.org/tensor/internal/utils"
)

func TestFoo(t *testing.T) {
	mul := func(a, b float64) float64 { return a * b }
	add := func(a, b float64) float64 { return a + b }

	m, k, n := 3, 2, 5
	a := gutils.Range[float64](0, m*k) // 3, 2
	b := gutils.Range[float64](0, k*n) // 2, 5
	c := make([]float64, m*n)          // 3, 5
	t.Logf("a %v\nb %v\nc %v", a, b, c)

	lda, ldb, ldc := k, n, n
	for i := 0; i < m; i++ {
		ctmp := c[i*ldc : i*ldc+n]
		idx := i * lda
		r := a[idx : idx+k]
		for l, v := range r {

			b2 := b[l*ldb : l*ldb+n]
			Afxry(mul, add, v, b2, ctmp)
		}
	}

	// impl := gonum.Implementation{}
	// impl.Dgemm(blas.NoTrans, blas.NoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc)
	// t.Logf("a %v\nb %v\nc %v", a, b, c)

	t.Logf("a %v\nb %v\nc %v", a, b, c)

}
