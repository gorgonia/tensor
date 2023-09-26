package stdeng_test

import (
	"context"
	"testing"

	"gorgonia.org/tensor/dense"
	. "gorgonia.org/tensor/engines"
	gutils "gorgonia.org/tensor/internal/utils"
	"github.com/stretchr/testify/assert"
)

func TestStdEng_Scatter(t *testing.T) {
	assert := assert.New(t)
	d := dense.New[float64](WithShape(2, 3), WithBacking(gutils.Range[float64](1, 7)))
	indices := dense.New[int](WithShape(2, 2), WithBacking([]int{1, 3, 2, 1}))
	retVal := dense.New[float64](WithShape(2, 4))

	t.Logf("\n%v", d)
	t.Logf("\n%v", indices)
	e := StdEng[float64, *dense.Dense[float64]]{}
	err := e.Scatter(context.Background(), d, indices, retVal)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("retVal \n%v", retVal)

	expected := dense.New[float64](WithShape(2, 4), WithBacking([]float64{
		0, 1, 0, 2,
		0, 5, 4, 0}))
	assert.Equal(expected.Data(), retVal.Data())
	assert.True(expected.Shape().Eq(retVal.Shape()))
}
