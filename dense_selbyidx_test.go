package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDense_SelectByIndices(t *testing.T) {
	assert := assert.New(t)

	a := New(WithBacking([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}), WithShape(3, 2, 4))
	indices := New(WithBacking([]int{1, 1}))

	e := StdEng{}

	a1, err := e.SelectByIndices(a, indices, 1)
	if err != nil {
		t.Errorf("%v", err)
	}
	correct1 := []float64{4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21, 22, 23}
	assert.Equal(correct1, a1.Data())

	a0, err := e.SelectByIndices(a, indices, 0)
	if err != nil {
		t.Errorf("%v", err)
	}
	correct0 := []float64{8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15}
	assert.Equal(correct0, a0.Data())

	a2, err := e.SelectByIndices(a, indices, 2)
	if err != nil {
		t.Errorf("%v", err)
	}
	correct2 := []float64{1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21}
	assert.Equal(correct2, a2.Data())

	// !safe
	aUnsafe := a.Clone().(*Dense)
	indices = New(WithBacking([]int{1, 1, 1}))
	aUnsafeSelect, err := e.SelectByIndices(aUnsafe, indices, 0, UseUnsafe())
	if err != nil {
		t.Errorf("%v", err)
	}
	correctUnsafe := []float64{8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15}
	assert.Equal(correctUnsafe, aUnsafeSelect.Data())

	// 3 indices, just to make sure the sanity of the algorithm
	indices = New(WithBacking([]int{1, 1, 1}))
	a1, err = e.SelectByIndices(a, indices, 1)
	if err != nil {
		t.Errorf("%v", err)
	}
	correct1 = []float64{
		4, 5, 6, 7,
		4, 5, 6, 7,
		4, 5, 6, 7,

		12, 13, 14, 15,
		12, 13, 14, 15,
		12, 13, 14, 15,

		20, 21, 22, 23,
		20, 21, 22, 23,
		20, 21, 22, 23,
	}
	assert.Equal(correct1, a1.Data())

	a0, err = e.SelectByIndices(a, indices, 0)
	if err != nil {
		t.Errorf("%v", err)
	}
	correct0 = []float64{8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15}
	assert.Equal(correct0, a0.Data())

	a2, err = e.SelectByIndices(a, indices, 2)
	if err != nil {
		t.Errorf("%v", err)
	}
	correct2 = []float64{1, 1, 1, 5, 5, 5, 9, 9, 9, 13, 13, 13, 17, 17, 17, 21, 21, 21}
	assert.Equal(correct2, a2.Data())
}
