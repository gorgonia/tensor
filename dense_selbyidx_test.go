package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var selByIndicesTests = []struct {
	Name    string
	Data    interface{}
	Shape   Shape
	Indices []int
	Axis    int
	WillErr bool

	Correct      interface{}
	CorrectShape Shape
}{
	{Name: "3-tensor, axis 0", Data: Range(Float64, 0, 24), Shape: Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 0, WillErr: false,
		Correct: []float64{8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15}, CorrectShape: Shape{2, 2, 4}},

	{Name: "3-tensor, axis 1", Data: Range(Float64, 0, 24), Shape: Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 1, WillErr: false,
		Correct: []float64{4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21, 22, 23}, CorrectShape: Shape{3, 2, 4}},

	{Name: "3-tensor, axis 2", Data: Range(Float64, 0, 24), Shape: Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 2, WillErr: false,
		Correct: []float64{1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21}, CorrectShape: Shape{3, 2, 2}},

	{Name: "Vector, axis 0", Data: Range(Int, 0, 5), Shape: Shape{5}, Indices: []int{1, 1}, Axis: 0, WillErr: false,
		Correct: []int{1, 1}, CorrectShape: Shape{2}},

	{Name: "Vector, axis 1", Data: Range(Int, 0, 5), Shape: Shape{5}, Indices: []int{1, 1}, Axis: 1, WillErr: true,
		Correct: []int{1, 1}, CorrectShape: Shape{2}},
}

func TestDense_SelectByIndices(t *testing.T) {
	assert := assert.New(t)
	for i, tc := range selByIndicesTests {
		T := New(WithShape(tc.Shape...), WithBacking(tc.Data))
		indices := New(WithBacking(tc.Indices))
		ret, err := ByIndices(T, indices, tc.Axis)
		if checkErr(t, tc.WillErr, err, tc.Name, i) {
			continue
		}
		assert.Equal(tc.Correct, ret.Data())
		assert.True(tc.CorrectShape.Eq(ret.Shape()))
	}
}

func TestDense_SelectByIndicesB(t *testing.T) {
	a := New(WithBacking([]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}), WithShape(3, 2, 4))
	indices := New(WithBacking([]int{1, 1}))

	t.Logf("a\n%v", a)

	e := StdEng{}

	a1, err := e.SelectByIndices(a, indices, 1)
	if err != nil {
		t.Errorf("%v", err)
	}
	t.Logf("a1\n%v", a1)

	a1Grad, err := e.SelectByIndicesB(a, a1, indices, 1)
	if err != nil {
		t.Errorf("%v", err)
	}
	t.Logf("a1Grad \n%v", a1Grad)

	a0, err := e.SelectByIndices(a, indices, 0)
	if err != nil {
		t.Errorf("%v", err)
	}
	t.Logf("a0\n%v", a0)
	a0Grad, err := e.SelectByIndicesB(a, a0, indices, 0)
	if err != nil {
		t.Errorf("%v", err)
	}
	t.Logf("a0Grad\n%v", a0Grad)

	a2, err := e.SelectByIndices(a, indices, 2)
	if err != nil {
		t.Errorf("%v", err)
	}
	t.Logf("\n%v", a2)
	a2Grad, err := e.SelectByIndicesB(a, a2, indices, 2)
	if err != nil {
		t.Errorf("%v", err)
	}
	t.Logf("a2Grad\n%v", a2Grad)
}
