package tensor

import (
	"testing"

	"golang.org/x/exp/constraints"
)

func addX[T constraints.Ordered](a, b T) (T, error) { return a + b, nil }
func TestPerv(t *testing.T) {
	// a := 1.0
	// b := New[float64](WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4}))
	// c, err := Perv[float64](add_[float64])(S(a), b)
	// t.Logf("c %v err %v", c.Data(), err)
}
