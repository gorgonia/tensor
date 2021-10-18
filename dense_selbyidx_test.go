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
	{Name: "Basic", Data: Range(Float64, 0, 4), Shape: Shape{2, 2}, Indices: []int{0, 1}, Axis: 0, WillErr: false,
		Correct: []float64{0, 1, 2, 3}, CorrectShape: Shape{2, 2},
	},
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
	{Name: "(2,1) Matrx (colvec) with (10) indices", Data: Range(Float64, 0, 2), Shape: Shape{2, 1}, Indices: []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, Axis: 0, WillErr: false,
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
	// Basic
	{
		CorrectGrad: []float64{1, 1, 1, 1},
	},
	// 3-tensor, axis 0
	{
		CorrectGrad: []float64{0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0},
	},
	// 3-tensor, axis 1
	{
		CorrectGrad: []float64{0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2},
	},
	// 3-tensor, axis 2
	{
		CorrectGrad: []float64{0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0},
	},
	// vector, axis 0
	{
		CorrectGrad: []int{0, 2, 0, 0, 0},
	},
	// vector, axis 1
	{
		CorrectGrad: []float32{4, 6, 8, 12, 8, 12, 0, 0},
	},
	// (4,2) Matrix with (10) indices
	{
		CorrectGrad: []float32{2, 2, 4, 4, 4, 4, 0, 0},
	},
	// (2, 1) Matrix (colvec) with (10) indices
	{
		CorrectGrad: []float64{0, 10},
	},
}

func init() {
	for i := range selByIndicesBTests {
		selByIndicesBTests[i].selByIndicesTest = selByIndicesTests[i]
		selByIndicesBTests[i].CorrectGradShape = selByIndicesTests[i].Shape
	}
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
		outGrad := ret.Clone().(*Dense)
		switch outGrad.Dtype() {
		case Float64:
			outGrad.Memset(1.0)
		case Float32:
			outGrad.Memset(float32(1.0))
		}

		grad, err := ByIndicesB(T, outGrad, indices, tc.Axis)
		if checkErr(t, tc.WillErr, err, tc.Name, i) {
			continue
		}
		assert.Equal(tc.CorrectGrad, grad.Data(), "%v - x:\n%v\nindices:\n%#v\ny:\n%#v\ngrad:\n%v", tc.Name, T, indices, ret, grad)
		assert.True(tc.CorrectGradShape.Eq(grad.Shape()), "%v - Grad shape should be %v. Got %v instead.\n\nx:\n%v\nindices:\n%#v\ny:\n%#v\ngrad:\n%v", tc.Name, tc.CorrectGradShape, grad.Shape(), T, indices, ret, grad)
	}

}
