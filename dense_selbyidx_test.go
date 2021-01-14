package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type selByIndicesTest struct {
	Name    string
	Data    interface{}
	Shape   Shape
	Indices []int
	Axis    int
	WillErr bool

	Correct      interface{}
	CorrectShape Shape
}

var selByIndicesTests = []selByIndicesTest{
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
	{Name: "(4,2) Matrix, with (10) indices", Data: Range(Float32, 0, 8), Shape: Shape{4, 2}, Indices: []int{1, 1, 1, 1, 0, 2, 2, 2, 2, 0}, Axis: 0, WillErr: false,
		Correct: []float32{2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 4, 5, 4, 5, 4, 5, 4, 5, 0, 1}, CorrectShape: Shape{10, 2}},
	{Name: "(2,1) Matrx (colvec)m with (10) indies", Data: Range(Float64, 0, 2), Shape: Shape{2, 1}, Indices: []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, Axis: 0, WillErr: false,
		Correct: []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, CorrectShape: Shape{10},
	},
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

var selByIndicesBTests = []struct {
	selByIndicesTest

	CorrectGrad      interface{}
	CorrectGradShape Shape
}{
	{
		selByIndicesTest: selByIndicesTests[0],
		CorrectGrad:      []float64{0, 0, 0, 0, 0, 0, 0, 0, 16, 18, 20, 22, 24, 26, 28, 30, 0, 0, 0, 0, 0, 0, 0, 0},
		CorrectGradShape: Shape{3, 2, 4},
	},
	{
		selByIndicesTest: selByIndicesTests[1],
		CorrectGrad:      []float64{0, 0, 0, 0, 8, 10, 12, 14, 0, 0, 0, 0, 24, 26, 28, 30, 0, 0, 0, 0, 40, 42, 44, 46},
		CorrectGradShape: Shape{3, 2, 4},
	},
	{
		selByIndicesTest: selByIndicesTests[2],
		CorrectGrad:      []float64{0, 2, 0, 0, 0, 10, 0, 0, 0, 18, 0, 0, 0, 26, 0, 0, 0, 34, 0, 0, 0, 42, 0, 0},
		CorrectGradShape: Shape{3, 2, 4},
	},
	{
		selByIndicesTest: selByIndicesTests[3],
		CorrectGrad:      []int{0, 2, 0, 0, 0},
		CorrectGradShape: Shape{5},
	},
	{
		selByIndicesTest: selByIndicesTests[5],
		CorrectGrad:      []float32{4, 6, 8, 12, 8, 12, 0, 0},
		CorrectGradShape: Shape{4, 2},
	},
	{
		selByIndicesTest: selByIndicesTests[6],
		CorrectGrad:      []float64{0, 10},
		CorrectGradShape: Shape{2, 1},
	},
}

func TestDense_SelectByIndicesB(t *testing.T) {

	assert := assert.New(t)
	for i, tc := range selByIndicesBTests {
		T := New(WithShape(tc.Shape...), WithBacking(tc.Data))
		indices := New(WithBacking(tc.Indices))
		ret, err := ByIndices(T, indices, tc.Axis)
		if checkErr(t, tc.WillErr, err, tc.Name, i) {
			continue
		}
		grad, err := ByIndicesB(T, ret, indices, tc.Axis)
		if checkErr(t, tc.WillErr, err, tc.Name, i) {
			continue
		}
		assert.Equal(tc.CorrectGrad, grad.Data(), "%v", tc.Name)
		assert.True(tc.CorrectGradShape.Eq(grad.Shape()), "%v - Grad shape should be %v. Got %v instead", tc.Name, tc.CorrectGradShape, grad.Shape())
	}

}
