package dense

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestDense_Lt(t *testing.T) {
	T := New[float64](WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4}))
	// U := New[float64](WithShape(2, 2), WithBacking([]float64{1, 3, 2, 4}))
	// v, err := T.Lt(U)
	// if err != nil {
	// 	t.Fatalf("err %v", err)
	// }
	// t.Logf("%v", v)

	U := New[float64](WithShape(2), WithBacking([]float64{0, 3}))
	v, err := T.Lt(U, tensor.AutoBroadcast)
	if err != nil {
		t.Fatalf("err %v", err)
	}
	t.Logf("%v", v)
}
