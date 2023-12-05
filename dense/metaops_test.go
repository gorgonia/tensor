package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal"
	gutils "gorgonia.org/tensor/internal/utils"
)

var denseSliceTests = []struct {
	name   string
	data   []float64
	shape  shapes.Shape
	slices []SliceRange

	correctShape   shapes.Shape
	correctStrides []int
	correctData    []float64
}{

	// scalar-equiv vector (issue 102)
	{"a[0], a is scalar-equiv", []float64{2},
		shapes.Shape{1}, []SliceRange{SR(0)}, shapes.ScalarShape(), nil, []float64{2}},

	// vector
	{"a[0]", []float64{1, 1, 0, 0, 0},
		shapes.Shape{5}, []SliceRange{SR(0)}, shapes.ScalarShape(), nil, []float64{1}},
	{"a[0:2]", gutils.Range[float64](0, 5), shapes.Shape{5}, []SliceRange{SR(0, 2)}, shapes.Shape{2}, []int{1}, []float64{0, 1}},
	{"a[1:5:2]", gutils.Range[float64](0, 5), shapes.Shape{5}, []SliceRange{SR(1, 5, 2)}, shapes.Shape{2}, []int{2}, []float64{1, 2, 3, 4}},

	// colvec
	{"c[0]", gutils.Range[float64](0, 5), shapes.Shape{5, 1}, []SliceRange{SR(0)}, shapes.ScalarShape(), nil, []float64{0}},
	{"c[0:2]", gutils.Range[float64](0, 5), shapes.Shape{5, 1}, []SliceRange{SR(0, 2)}, shapes.Shape{2, 1}, []int{1, 1}, []float64{0, 1}},
	{"c[1:5:2]", gutils.Range[float64](0, 5), shapes.Shape{5, 1}, []SliceRange{SR(0, 5, 2)}, shapes.Shape{2, 1}, []int{2, 1}, []float64{0, 1, 2, 3, 4}},

	// // rowvec
	{"r[0]", gutils.Range[float64](0, 5), shapes.Shape{1, 5}, []SliceRange{SR(0)}, shapes.Shape{1, 5}, []int{1}, []float64{0, 1, 2, 3, 4}},
	{"r[0:2]", gutils.Range[float64](0, 5), shapes.Shape{1, 5}, []SliceRange{SR(0, 2)}, shapes.Shape{1, 5}, []int{1}, []float64{0, 1, 2, 3, 4}},
	{"r[0:5:2]", gutils.Range[float64](0, 5), shapes.Shape{1, 5}, []SliceRange{SR(0, 5, 2)}, shapes.Shape{1, 5}, []int{1}, []float64{0, 1, 2, 3, 4}},
	{"r[:, 0]", gutils.Range[float64](0, 5), shapes.Shape{1, 5}, []SliceRange{nil, SR(0)}, shapes.ScalarShape(), nil, []float64{0}},
	{"r[:, 0:2]", gutils.Range[float64](0, 5), shapes.Shape{1, 5}, []SliceRange{nil, SR(0, 2)}, shapes.Shape{1, 2}, []int{5, 1}, []float64{0, 1}},
	{"r[:, 1:5:2]", gutils.Range[float64](0, 5), shapes.Shape{1, 5}, []SliceRange{nil, SR(1, 5, 2)}, shapes.Shape{1, 2}, []int{5, 2}, []float64{1, 2, 3, 4}},

	// // matrix
	{"A[0]", gutils.Range[float64](0, 6), shapes.Shape{2, 3}, []SliceRange{SR(0)}, shapes.Shape{1, 3}, []int{1}, gutils.Range[float64](0, 3)},
	{"A[0:2]", gutils.Range[float64](0, 20), shapes.Shape{4, 5}, []SliceRange{SR(0, 2)}, shapes.Shape{2, 5}, []int{5, 1}, gutils.Range[float64](0, 10)},
	{"A[0, 0]", gutils.Range[float64](0, 20), shapes.Shape{4, 5}, []SliceRange{SR(0), SR(0)}, shapes.ScalarShape(), nil, []float64{0}},
	{"A[0, 1:5]", gutils.Range[float64](0, 20), shapes.Shape{4, 5}, []SliceRange{SR(0), SR(1, 5)}, shapes.Shape{4}, []int{1}, gutils.Range[float64](1, 5)},
	{"A[0, 1:5:2]", gutils.Range[float64](0, 20), shapes.Shape{4, 5}, []SliceRange{SR(0), SR(1, 5, 2)}, shapes.Shape{1, 2}, []int{2}, gutils.Range[float64](1, 5)},
	{"A[:, 0]", gutils.Range[float64](0, 20), shapes.Shape{4, 5}, []SliceRange{nil, SR(0)}, shapes.Shape{4, 1}, []int{5}, gutils.Range[float64](0, 16)},
	{"A[:, 1:5]", gutils.Range[float64](0, 20), shapes.Shape{4, 5}, []SliceRange{nil, SR(1, 5)}, shapes.Shape{4, 4}, []int{5, 1}, gutils.Range[float64](1, 20)},
	{"A[:, 1:5:2]", gutils.Range[float64](0, 20), shapes.Shape{4, 5}, []SliceRange{nil, SR(1, 5, 2)}, shapes.Shape{4, 2}, []int{5, 2}, gutils.Range[float64](1, 20)},

	// 3tensor with leading and trailing 1s

	{"3T1[0]", gutils.Range[float64](0, 9), shapes.Shape{1, 9, 1}, []SliceRange{SR(0)}, shapes.Shape{9, 1}, []int{1, 1}, gutils.Range[float64](0, 9)},
	{"3T1[nil, 0:2]", gutils.Range[float64](0, 9), shapes.Shape{1, 9, 1}, []SliceRange{nil, SR(0, 2)}, shapes.Shape{1, 2, 1}, []int{9, 1, 1}, gutils.Range[float64](0, 2)},
	{"3T1[nil, 0:5:3]", gutils.Range[float64](0, 9), shapes.Shape{1, 9, 1}, []SliceRange{nil, SR(0, 5, 3)}, shapes.Shape{1, 2, 1}, []int{9, 3, 1}, gutils.Range[float64](0, 5)},
	{"3T1[nil, 1:5:3]", gutils.Range[float64](0, 9), shapes.Shape{1, 9, 1}, []SliceRange{nil, SR(1, 5, 3)}, shapes.Shape{1, 2, 1}, []int{9, 3, 1}, gutils.Range[float64](1, 5)},
	{"3T1[nil, 1:9:3]", gutils.Range[float64](0, 9), shapes.Shape{1, 9, 1}, []SliceRange{nil, SR(1, 9, 3)}, shapes.Shape{1, 3, 1}, []int{9, 3, 1}, gutils.Range[float64](1, 9)},

	// 3tensor
	{"3T[0]", gutils.Range[float64](0, 36), shapes.Shape{2, 9, 2}, []SliceRange{SR(0)}, shapes.Shape{9, 2}, []int{2, 1}, gutils.Range[float64](0, 18)},
	{"3T[1]", gutils.Range[float64](0, 36), shapes.Shape{2, 9, 2}, []SliceRange{SR(1)}, shapes.Shape{9, 2}, []int{2, 1}, gutils.Range[float64](18, 36)},
	{"3T[1, 2]", gutils.Range[float64](0, 36), shapes.Shape{2, 9, 2}, []SliceRange{SR(1), SR(2)}, shapes.Shape{2}, []int{1}, gutils.Range[float64](22, 24)},
	{"3T[1, 2:4]", gutils.Range[float64](0, 36), shapes.Shape{2, 9, 2}, []SliceRange{SR(1), SR(2, 4)}, shapes.Shape{2, 2}, []int{2, 1}, gutils.Range[float64](22, 26)},
	{"3T[1, 2:8:2]", gutils.Range[float64](0, 36), shapes.Shape{2, 9, 2}, []SliceRange{SR(1), SR(2, 8, 2)}, shapes.Shape{3, 2}, []int{4, 1}, gutils.Range[float64](22, 34)},
	{"3T[1, 2:8:3]", gutils.Range[float64](0, 36), shapes.Shape{2, 9, 2}, []SliceRange{SR(1), SR(2, 8, 3)}, shapes.Shape{2, 2}, []int{6, 1}, gutils.Range[float64](22, 34)},
	{"3T[1, 2:9:2]", gutils.Range[float64](0, 126), shapes.Shape{2, 9, 7}, []SliceRange{SR(1), SR(2, 9, 2)}, shapes.Shape{4, 7}, []int{14, 1}, gutils.Range[float64](77, 126)},
	{"3T[1, 2:9:2, 1]", gutils.Range[float64](0, 126), shapes.Shape{2, 9, 7}, []SliceRange{SR(1), SR(2, 9, 2), SR(1)}, shapes.Shape{4}, []int{14}, gutils.Range[float64](78, 121)}, // should this be a colvec?
	{"3T[1, 2:9:2, 1:4:2]", gutils.Range[float64](0, 126), shapes.Shape{2, 9, 7}, []SliceRange{SR(1), SR(2, 9, 2), SR(1, 4, 2)}, shapes.Shape{4, 2}, []int{14, 2}, gutils.Range[float64](78, 123)},
}

func TestDense_Slice(t *testing.T) {
	assert := assert.New(t)
	for _, test := range denseSliceTests {
		t.Run(test.name, func(t *testing.T) {
			a := New[float64](WithShape(test.shape...), WithBacking(test.data))
			b, err := a.Slice(test.slices...)
			assert.NoError(err)
			assert.Equal(test.correctShape, b.Shape())
			assert.Equal(test.correctStrides, b.Strides())
			assert.Equal(test.correctData, b.Data())

		})
	}

	// Test slicing a sliced tensor
	a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
	b, err := a.Slice(SR(0, 1))
	assert.NoError(err)
	c, err := b.Slice(SR(2, 3))
	assert.NoError(err)
	assert.Equal(shapes.Shape{4}, c.Shape())
	assert.Equal([]int{1}, c.Strides())
	assert.Equal([]float64{8, 9, 10, 11}, c.Data())

}

var rollaxisTests = []struct {
	axis, start int

	correctShape shapes.Shape
}{
	{0, 0, shapes.Shape{1, 2, 3, 4}},
	{0, 1, shapes.Shape{1, 2, 3, 4}},
	{0, 2, shapes.Shape{2, 1, 3, 4}},
	{0, 3, shapes.Shape{2, 3, 1, 4}},
	{0, 4, shapes.Shape{2, 3, 4, 1}},

	{1, 0, shapes.Shape{2, 1, 3, 4}},
	{1, 1, shapes.Shape{1, 2, 3, 4}},
	{1, 2, shapes.Shape{1, 2, 3, 4}},
	{1, 3, shapes.Shape{1, 3, 2, 4}},
	{1, 4, shapes.Shape{1, 3, 4, 2}},

	{2, 0, shapes.Shape{3, 1, 2, 4}},
	{2, 1, shapes.Shape{1, 3, 2, 4}},
	{2, 2, shapes.Shape{1, 2, 3, 4}},
	{2, 3, shapes.Shape{1, 2, 3, 4}},
	{2, 4, shapes.Shape{1, 2, 4, 3}},

	{3, 0, shapes.Shape{4, 1, 2, 3}},
	{3, 1, shapes.Shape{1, 4, 2, 3}},
	{3, 2, shapes.Shape{1, 2, 4, 3}},
	{3, 3, shapes.Shape{1, 2, 3, 4}},
	{3, 4, shapes.Shape{1, 2, 3, 4}},
}

// The RollAxis tests are directly adapted from Numpy's test cases.
func TestDense_RollAxis(t *testing.T) {
	assert := assert.New(t)
	var T *Dense[int]
	var err error

	for _, rats := range rollaxisTests {
		T = New[int](WithShape(1, 2, 3, 4))
		if _, err = T.RollAxis(rats.axis, rats.start, false); assert.NoError(err) {
			assert.True(rats.correctShape.Eq(T.Shape()), "%d %d Expected %v, got %v", rats.axis, rats.start, rats.correctShape, T.Shape())
		}
	}
}

var transposeTests = []struct {
	name          string
	shape         shapes.Shape
	transposeWith []int
	data          []float64

	correctShape    shapes.Shape
	correctStrides  []int // after .T()
	correctStrides2 []int // after .Transpose()
	correctData     []float64
}{
	{"c.T()", shapes.Shape{4, 1}, nil, []float64{0, 1, 2, 3},
		shapes.Shape{1, 4}, []int{1, 1}, []int{4, 1}, []float64{0, 1, 2, 3}},

	{"r.T()", shapes.Shape{1, 4}, nil, []float64{0, 1, 2, 3},
		shapes.Shape{4, 1}, []int{1, 1}, []int{1, 1}, []float64{0, 1, 2, 3}},

	{"v.T()", shapes.Shape{4}, nil, []float64{0, 1, 2, 3},
		shapes.Shape{4}, []int{1}, []int{1}, []float64{0, 1, 2, 3}},

	{"M.T()", shapes.Shape{2, 3}, nil, []float64{0, 1, 2, 3, 4, 5},
		shapes.Shape{3, 2}, []int{1, 3}, []int{2, 1}, []float64{0, 3, 1, 4, 2, 5}},

	{"M.T(0,1) (NOOP)", shapes.Shape{2, 3}, []int{0, 1}, []float64{0, 1, 2, 3, 4, 5},
		shapes.Shape{2, 3}, []int{3, 1}, []int{3, 1}, []float64{0, 1, 2, 3, 4, 5}},

	{"3T.T()", shapes.Shape{2, 3, 4}, nil,
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},

		shapes.Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1},
		[]float64{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},

	{"3T.T(2, 1, 0) (Same as .T())", shapes.Shape{2, 3, 4}, []int{2, 1, 0},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		shapes.Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1},
		[]float64{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},

	{"3T.T(2, 1, 0) (Same as .T())", shapes.Shape{2, 3, 4}, []int{2, 1, 0},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		shapes.Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1},
		[]float64{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},

	{"3T.T(0, 2, 1)", shapes.Shape{2, 3, 4}, []int{0, 2, 1},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		shapes.Shape{2, 4, 3}, []int{12, 1, 4}, []int{12, 3, 1},
		[]float64{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}},

	{"3T.T{1, 0, 2)", shapes.Shape{2, 3, 4}, []int{1, 0, 2},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		shapes.Shape{3, 2, 4}, []int{4, 12, 1}, []int{8, 4, 1},
		[]float64{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}},

	{"3T.T{1, 2, 0)", shapes.Shape{2, 3, 4}, []int{1, 2, 0},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		shapes.Shape{3, 4, 2}, []int{4, 1, 12}, []int{8, 2, 1},
		[]float64{0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23}},

	{"3T.T{2, 0, 1)", shapes.Shape{2, 3, 4}, []int{2, 0, 1},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		shapes.Shape{4, 2, 3}, []int{1, 12, 4}, []int{6, 3, 1},
		[]float64{0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}},

	{"3T.T{0, 1, 2} (NOOP)", shapes.Shape{2, 3, 4}, []int{0, 1, 2},
		[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0},
		shapes.Shape{2, 3, 4}, []int{12, 4, 1}, []int{12, 4, 1},
		[]float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}},

	///////

	// {"M[2,2].T for bools, just for completeness sake", shapes.Shape{2, 2}, nil,
	// 	[]bool{true, true, false, false},
	// 	shapes.Shape{2, 2}, []int{1, 2}, []int{2, 1},
	// 	[]bool{true, false, true, false},
	// },

	// {"M[2,2].T for strings, just for completeness sake", shapes.Shape{2, 2}, nil,
	// 	[]string{"hello", "world", "今日は", "世界"},
	// 	shapes.Shape{2, 2}, []int{1, 2}, []int{2, 1},
	// 	[]string{"hello", "今日は", "world", "世界"},
	// },
}

func TestDense_Transpose(t *testing.T) {
	assert := assert.New(t)
	var err error

	// standard transposes
	for _, tts := range transposeTests {
		T := New[float64](WithShape(tts.shape...), WithBacking(tts.data))
		var T2, T3 *Dense[float64]
		if T2, err = T.T(tts.transposeWith...); err != nil {
			if err = internal.HandleNoOp(err); err != nil {
				t.Errorf("%v - %v", tts.name, err)
			}
			continue
		}
		assert.True(tts.correctShape.Eq(T2.Shape()), ".T() %v Expected shape: %v. Got %v", tts.name, tts.correctShape, T2.Shape())
		assert.Equal(tts.correctStrides, T2.Strides(), ".T() %v. Expected stride: %v. Got %v", tts.name, tts.correctStrides, T2.Strides())
		assert.Equal(tts.data, T2.Data(), ".T() %v", tts.name)

		if T3, err = T.Transpose(tts.transposeWith...); err != nil {
			if err = internal.HandleNoOp(err); err != nil {
				t.Errorf("%v - %v", tts.name, err)
			}
			continue
		}

		assert.True(tts.correctShape.Eq(T3.Shape()), ".Transpose() %v Expected shape: %v. Got %v", tts.name, tts.correctShape, T3.Shape())
		assert.Equal(tts.correctStrides2, T3.Strides(), ".Transpose() %v - Expected stride %v. Got %v", tts.name, tts.correctStrides2, T3.Strides())
		assert.Equal(tts.correctData, T3.Data(), ".Transpose() %v", tts.name)
	}

	// 	// test stacked .T() calls
	// 	var T *Dense

	// 	// column vector
	// 	T = New(WithShape(4, 1), WithBacking(Range(Int, 0, 4)))
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #1 for vector. Error: %v", err)
	// 		goto matrev
	// 	}
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #1 for vector. Error: %v", err)
	// 		goto matrev
	// 	}
	// 	assert.True(T.old.IsZero())
	// 	assert.Nil(T.transposeWith)
	// 	assert.True(T.IsColVec())

	// matrev:
	// 	// matrix, reversed
	// 	T = New(WithShape(2, 3), WithBacking(Range(Byte, 0, 6)))
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #1 for matrix reverse. Error: %v", err)
	// 		goto matnorev
	// 	}
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #2 for matrix reverse. Error: %v", err)
	// 		goto matnorev
	// 	}
	// 	assert.True(T.old.IsZero())
	// 	assert.Nil(T.transposeWith)
	// 	assert.True(Shape{2, 3}.Eq(T.Shape()))

	// matnorev:
	// 	// 3-tensor, non reversed
	// 	T = New(WithShape(2, 3, 4), WithBacking(Range(Int64, 0, 24)))
	// 	if err = T.T(); err != nil {
	// 		t.Fatalf("Stacked .T() #1 for tensor with no reverse. Error: %v", err)
	// 	}
	// 	if err = T.T(2, 0, 1); err != nil {
	// 		t.Fatalf("Stacked .T() #2 for tensor with no reverse. Error: %v", err)
	// 	}
	// 	correctData := []int64{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}
	// 	assert.Equal(correctData, T.Data())
	// 	assert.Equal([]int{2, 0, 1}, T.transposeWith)
	// 	assert.NotNil(T.old)

}
