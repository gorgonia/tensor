package dense

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/chewxy/inigo/values/tensor/internal/axialiter"
	"github.com/chewxy/inigo/values/tensor/internal/execution"
	gutils "github.com/chewxy/inigo/values/tensor/internal/utils"
	"gorgonia.org/dtype"
)

func TestNewDenseOfDifferentType(t *testing.T) {
	d := New[float64](WithBacking([]float64{0, 1, 2, 3}))
	dbool := d.AlikeAsType(dtype.Bool, WithBacking([]bool{false, true, true, true}))
	t.Logf("dBool %v", dbool)
}

func TestIter(t *testing.T) {
	T := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 2*3*4)))
	t.Logf("T\n%v", T)
	T2, err := T.T(0, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("T2\n%v", T2)
	fit := T2.Iterator()
	indices, err := execution.ArgmaxIter(T.Data(), fit, 3)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("indices %v", indices)

	it := axialiter.New(T.Info(), 2, 4, true)
	var buf bytes.Buffer
	for next, err := it.Next(); err == nil; next, err = it.Next() {
		fmt.Fprintf(&buf, "%v, ", T.Data()[next])
	}
	t.Logf("%v", buf.String())
}
