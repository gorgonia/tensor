package tensor

import "testing"

func Test_conv(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	b := cast[float64, float32](a)
	t.Logf("a %v, b %v", a, b)
	t.Logf("len(b) %d cap(b) %d", len(b), cap(b))
}
