package dense

import (
	"testing"

	gutils "gorgonia.org/tensor/internal/utils"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

func TestDense_Reduce(t *testing.T) {
	assert := assert.New(t)
	t.Run("Basic", func(t *testing.T) {
		a := New[int](WithBacking([]int{1, 2, 3, 4, 5}))
		expected := New[int](WithShape(), WithBacking([]int{15}))

		sum, err := a.Reduce(func(a, b int) int { return a + b }, 0)
		if err != nil {
			t.Errorf("Reduce failed with error: %v", err)
		}
		assert.True(expected.Shape().Eq(sum.Shape()))
		assert.Equal(expected.ScalarValue(), sum.ScalarValue())
	})

	t.Run("Reduce dim 0 of matrix", func(t *testing.T) {
		a := New[float32](WithShape(2, 3), WithBacking([]float32{1, 2, 3, 4, 5, 6}))
		expected := New[float32](WithShape(3), WithBacking([]float32{5, 7, 9}))

		sum, err := a.Reduce(func(a, b float32) float32 { return a + b }, 0.0, Along(0))
		if err != nil {
			t.Errorf("Reduce failed with error: %v", err)
		}
		assert.True(expected.Shape().Eq(sum.Shape()))
		assert.Equal(expected.Data(), sum.Data())
	})

	t.Run("Reduce dim 1 of matrix", func(t *testing.T) {
		a := New[float32](WithShape(2, 3), WithBacking([]float32{1, 2, 3, 4, 5, 6}))
		expected := New[float32](WithShape(2), WithBacking([]float32{6, 15}))

		sum, err := a.Reduce(func(a, b float32) float32 { return a + b }, 0.0, Along(1))
		if err != nil {
			t.Errorf("Reduce failed with error: %v", err)
		}
		assert.True(expected.Shape().Eq(sum.Shape()))
		assert.Equal(expected.Data(), sum.Data())
	})

	t.Run("Reduce all dims of matrix", func(t *testing.T) {
		a := New[float32](WithShape(2, 3), WithBacking([]float32{1, 2, 3, 4, 5, 6}))
		expected := New[float32](WithShape(), WithBacking([]float32{21}))

		sum, err := a.Reduce(func(a, b float32) float32 { return a + b }, 0.0)
		if err != nil {
			t.Errorf("Reduce failed with error: %v", err)
		}
		assert.True(expected.Shape().Eq(sum.Shape()))
		assert.Equal(expected.Data(), sum.Data())
	})
}

func TestDense_Dot(t *testing.T) {
	assert := assert.New(t)
	t.Run("Basic Sanity Check", func(t *testing.T) {
		a := New[float64](WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4}))
		b := New[float64](WithShape(2, 3), WithBacking([]float64{6, 5, 4, 3, 2, 1}))
		add := func(a, b float64) float64 { return a + b }
		mul := func(a, b float64) float64 { return a * b }
		c, err := a.Dot(add, mul, b)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("c\n%v", c)

		// sanity check
		c2, err := a.MatMul(b)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("c2\n%v", c2)
		assert.Equal(c2, c)
	})

}

type repeatTest[DT any] struct {
	name    string
	tensor  *Dense[DT]
	ne      bool // should assert tensor not equal
	axis    int
	repeats []int

	correct []DT
	shape   shapes.Shape
	err     bool
}

func makeRepeatTests[DT interface{ int | float32 | float64 }]() []repeatTest[DT] {
	return []repeatTest[DT]{
		{"Scalar Repeat on axis 0", New[DT](FromScalar(DT(1))),
			true, 0, []int{3},
			[]DT{1, 1, 1},
			shapes.Shape{3}, false,
		},

		{"Scalar Repeat on axis 1", New[DT](FromScalar(DT(255))),
			false, 1, []int{3},
			[]DT{255, 255, 255},
			shapes.Shape{1, 3}, false,
		},

		{"Scalar Repeat on axis 4", New[DT](FromScalar(DT(255))),
			false, 4, []int{3},
			[]DT{255, 255, 255},
			shapes.Shape{1, 1, 1, 1, 3}, false,
		},

		{"Vector Repeat on axis 0", New[DT](WithShape(2), WithBacking(gutils.Range[DT](1, 3))),
			false, 0, []int{3},
			[]DT{1, 1, 1, 2, 2, 2},
			shapes.Shape{6}, false,
		},

		{"ColVec Repeat on axis 0", New[DT](WithShape(2, 1), WithBacking(gutils.Range[DT](1, 3))),
			false, 0, []int{3},
			[]DT{1, 1, 1, 2, 2, 2},
			shapes.Shape{6, 1}, false,
		},

		{"RowVec Repeat on axis 0", New[DT](WithShape(1, 2), WithBacking(gutils.Range[DT](1, 3))),
			false, 0, []int{3},
			[]DT{1, 2, 1, 2, 1, 2},
			shapes.Shape{3, 2}, false,
		},

		{"ColVec Repeat on axis 1", New[DT](WithShape(2, 1), WithBacking(gutils.Range[DT](1, 3))),
			false, 1, []int{3},
			[]DT{1, 1, 1, 2, 2, 2},
			shapes.Shape{2, 3}, false,
		},

		{"RowVec Repeat on axis 1", New[DT](WithShape(1, 2), WithBacking(gutils.Range[DT](1, 3))),
			false, 1, []int{3},
			[]DT{1, 1, 1, 2, 2, 2},
			shapes.Shape{1, 6}, false,
		},

		{"Vector Repeat on all axes", New[DT](WithShape(2), WithBacking(gutils.Range[DT](1, 3))),
			false, AllAxes, []int{3},
			[]DT{1, 1, 1, 2, 2, 2},
			shapes.Shape{6}, false,
		},

		{"ColVec Repeat on all axes", New[DT](WithShape(2, 1), WithBacking(gutils.Range[DT](1, 3))),
			false, AllAxes, []int{3},
			[]DT{1, 1, 1, 2, 2, 2},
			shapes.Shape{6}, false,
		},

		{"RowVec Repeat on all axes", New[DT](WithShape(1, 2), WithBacking(gutils.Range[DT](1, 3))),
			false, AllAxes, []int{3},
			[]DT{1, 1, 1, 2, 2, 2},
			shapes.Shape{6}, false,
		},

		{"M[2,2] Repeat on all axes with repeats = (1,2,1,1)", New[DT](WithShape(2, 2), WithBacking(gutils.Range[DT](1, 5))),
			false, AllAxes, []int{1, 2, 1, 1},
			[]DT{1, 2, 2, 3, 4},
			shapes.Shape{5}, false,
		},

		{"M[2,2] Repeat on axis 1 with repeats = (2, 1)", New[DT](WithShape(2, 2), WithBacking(gutils.Range[DT](1, 5))),
			false, 1, []int{2, 1},
			[]DT{1, 1, 2, 3, 3, 4},
			shapes.Shape{2, 3}, false,
		},

		{"M[2,2] Repeat on axis 1 with repeats = (1, 2)", New[DT](WithShape(2, 2), WithBacking(gutils.Range[DT](1, 5))),
			false, 1, []int{1, 2},
			[]DT{1, 2, 2, 3, 4, 4},
			shapes.Shape{2, 3}, false,
		},

		{"M[2,2] Repeat on axis 0 with repeats = (1, 2)", New[DT](WithShape(2, 2), WithBacking([]DT{1, 2, 3, 4})),
			false, 0, []int{1, 2},
			[]DT{1, 2, 3, 4, 3, 4},
			shapes.Shape{3, 2}, false,
		},

		{"M[2,2] Repeat on axis 0 with repeats = (2, 1)", New[DT](WithShape(2, 2), WithBacking([]DT{1, 2, 3, 4})),
			false, 0, []int{2, 1},
			[]DT{1, 2, 1, 2, 3, 4},
			shapes.Shape{3, 2}, false,
		},

		{"3T[2,3,2] Repeat on axis 1 with repeats = (1,2,1)", New[DT](WithShape(2, 3, 2), WithBacking(gutils.Range[DT](1, 2*3*2+1))),
			false, 1, []int{1, 2, 1},
			[]DT{1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12},
			shapes.Shape{2, 4, 2}, false,
		},

		{"3T[2,3,2] Generic Repeat by 2", New[DT](WithShape(2, 3, 2), WithBacking(gutils.Range[DT](1, 2*3*2+1))),
			false, AllAxes, []int{2},
			[]DT{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12},
			shapes.Shape{24}, false,
		},

		{"3T[2,3,2] repeat with broadcast errors", New[DT](WithShape(2, 3, 2), WithBacking(gutils.Range[DT](1, 2*3*2+1))),
			false, 0, []int{1, 2, 1},
			nil, nil, true,
		},

		// idiots
		{"Nonexistent axis", New[DT](WithShape(2, 1), WithBacking([]DT{1, 0})),
			false, 2, []int{3}, nil, nil, true,
		},
	}
}

func TestDense_Repeat(t *testing.T) {
	assert := assert.New(t)

	for i, test := range makeRepeatTests[float64]() {
		T, err := test.tensor.Repeat(test.axis, test.repeats...)
		if checkErr(t, test.err, err, "Repeat", i) {
			continue
		}

		if test.ne {
			assert.NotEqual(test.tensor, T, test.name)
		}

		assert.Equal(test.correct, T.Data(), test.name)
		assert.Equal(test.shape, T.Shape(), test.name)
	}

	for i, test := range makeRepeatTests[float32]() {
		T, err := test.tensor.Repeat(test.axis, test.repeats...)
		if checkErr(t, test.err, err, "Repeat", i) {
			continue
		}

		if test.ne {
			assert.NotEqual(test.tensor, T, test.name)
		}

		assert.Equal(test.correct, T.Data(), test.name)
		assert.Equal(test.shape, T.Shape(), test.name)
	}
}

// func TestDense_Repeat_Slow(t *testing.T) {
// 	rt2 := make([]repeatTest, len(repeatTests))
// 	for i, rt := range repeatTests {
// 		rt2[i] = repeatTest{
// 			name:    rt.name,
// 			ne:      rt.ne,
// 			axis:    rt.axis,
// 			repeats: rt.repeats,
// 			correct: rt.correct,
// 			shape:   rt.shape,
// 			err:     rt.err,
// 			tensor:  rt.tensor.Clone().(*Dense),
// 		}
// 	}
// 	for i := range rt2 {
// 		maskLen := rt2[i].tensor.len()
// 		mask := make([]bool, maskLen)
// 		rt2[i].tensor.mask = mask
// 	}

// 	assert := assert.New(t)

// 	for i, test := range rt2 {
// 		T, err := test.tensor.Repeat(test.axis, test.repeats...)
// 		if checkErr(t, test.err, err, "Repeat", i) {
// 			continue
// 		}

// 		var D DenseTensor
// 		if D, err = getDenseTensor(T); err != nil {
// 			t.Errorf("Expected Repeat to return a *Dense. got %v of %T instead", T, T)
// 			continue
// 		}

// 		if test.ne {
// 			assert.NotEqual(test.tensor, D, test.name)
// 		}

// 		assert.Equal(test.correct, D.Data(), test.name)
// 		assert.Equal(test.shape, D.Shape(), test.name)
// 	}
// }
