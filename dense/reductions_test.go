package dense

import (
	"testing"

	"gorgonia.org/tensor"
	gutils "gorgonia.org/tensor/internal/utils"
)

func TestDense_Sum(t *testing.T) {
	d := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
	res, err := d.Sum(tensor.Along(1), tensor.KeepDims)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v\n%v", res.Shape(), res)

	res, err = d.Sum(tensor.Along(1))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v\n%v", res.Shape(), res)
}
