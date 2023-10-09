package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
	gutils "gorgonia.org/tensor/internal/utils"
)

type selByIndicesTest[DT any] struct {
	Name    string
	Data    []DT
	Shape   shapes.Shape
	Indices []int
	Axis    int
	WillErr bool

	Correct      []DT
	CorrectShape shapes.Shape
}

func makeSelByIndicesTest[DT gutils.Rangeable]() []selByIndicesTest[DT] {
	return []selByIndicesTest[DT]{
		{Name: "Basic", Data: gutils.Range[DT](0, 4), Shape: shapes.Shape{2, 2}, Indices: []int{0, 1}, Axis: 0, WillErr: false,
			Correct: []DT{0, 1, 2, 3}, CorrectShape: shapes.Shape{2, 2},
		},
		{Name: "3-tensor, axis 0", Data: gutils.Range[DT](0, 24), Shape: shapes.Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 0, WillErr: false,
			Correct: []DT{8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15}, CorrectShape: shapes.Shape{2, 2, 4}},

		{Name: "3-tensor, axis 1", Data: gutils.Range[DT](0, 24), Shape: shapes.Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 1, WillErr: false,
			Correct: []DT{4, 5, 6, 7, 4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15, 20, 21, 22, 23, 20, 21, 22, 23}, CorrectShape: shapes.Shape{3, 2, 4}},

		{Name: "3-tensor, axis 2", Data: gutils.Range[DT](0, 24), Shape: shapes.Shape{3, 2, 4}, Indices: []int{1, 1}, Axis: 2, WillErr: false,
			Correct: []DT{1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21}, CorrectShape: shapes.Shape{3, 2, 2}},

		{Name: "Vector, axis 0", Data: gutils.Range[DT](0, 5), Shape: shapes.Shape{5}, Indices: []int{1, 1}, Axis: 0, WillErr: false,
			Correct: []DT{1, 1}, CorrectShape: shapes.Shape{2}},

		{Name: "Vector, axis 1", Data: gutils.Range[DT](0, 5), Shape: shapes.Shape{5}, Indices: []int{1, 1}, Axis: 1, WillErr: true,
			Correct: nil, CorrectShape: nil},
		{Name: "(4,2) Matrix, with (10) indices", Data: gutils.Range[DT](0, 8), Shape: shapes.Shape{4, 2}, Indices: []int{1, 1, 1, 1, 0, 2, 2, 2, 2, 0}, Axis: 0, WillErr: false,
			Correct: []DT{2, 3, 2, 3, 2, 3, 2, 3, 0, 1, 4, 5, 4, 5, 4, 5, 4, 5, 0, 1}, CorrectShape: shapes.Shape{10, 2}},
		{Name: "(2,1) Matrx (colvec) with (10) indices", Data: gutils.Range[DT](0, 2), Shape: shapes.Shape{2, 1}, Indices: []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, Axis: 0, WillErr: false,
			Correct: []DT{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, CorrectShape: shapes.Shape{10},
		},

		{Name: "ScalarEquiv, good", Data: []DT{15}, Shape: shapes.Shape{1, 1, 1}, Indices: []int{0, 0}, Axis: 1, WillErr: false,
			Correct: []DT{15}, CorrectShape: shapes.Shape{}},
		{Name: "ScalarEquiv, good", Data: []DT{15}, Shape: shapes.Shape{1, 1, 1}, Indices: []int{1, 0}, Axis: 1, WillErr: true,
			Correct: nil, CorrectShape: nil},
	}
}

func TestDense_SelectByIndices(t *testing.T) {
	assert := assert.New(t)
	for i, tc := range makeSelByIndicesTest[int]() {
		T := New[int](WithShape(tc.Shape...), WithBacking(tc.Data))
		indices := New[int](WithBacking(tc.Indices))
		ret, err := T.ByIndices(indices, tc.Axis)
		if checkErr(t, tc.WillErr, err, tc.Name, i) {
			continue
		}
		assert.Equal(tc.Correct, ret.Data())
		assert.True(tc.CorrectShape.Eq(ret.Shape()), "%v. Wanted %v. Got %v", tc.Name, tc.CorrectShape, ret.Shape())
	}

	for i, tc := range makeSelByIndicesTest[float64]() {
		T := New[float64](WithShape(tc.Shape...), WithBacking(tc.Data))
		indices := New[int](WithBacking(tc.Indices))
		ret, err := T.ByIndices(indices, tc.Axis)
		if checkErr(t, tc.WillErr, err, tc.Name, i) {
			continue
		}
		assert.Equal(tc.Correct, ret.Data())
		assert.True(tc.CorrectShape.Eq(ret.Shape()))
	}
}
