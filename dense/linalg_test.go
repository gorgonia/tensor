package dense

import (
	"fmt"
	"testing"

	gutils "gorgonia.org/tensor/internal/utils"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

var traceTests = []struct {
	data []int

	correct int
	err     bool
}{
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
	{[]int{0, 1, 2, 3, 4, 5}, 4, false},
}

func TestDense_Trace(t *testing.T) {
	for i, tts := range traceTests {
		T := New[int](WithBacking(tts.data), WithShape(2, 3))
		trace, err := T.Trace()
		if err != nil {
			t.Errorf("Err in case %d: %v", i, err)
			continue
		}
		if trace != tts.correct {
			t.Errorf("Expcted %v. Got %v as trace instead", tts.correct, trace)
		}

		//
		T = New[int](WithBacking(tts.data))
		_, err = T.Trace()
		if err == nil {
			t.Error("Expected an error when Trace() on non-matrices")
		}
	}
}

var innerTestsFloat64 = []struct {
	a, b           []float64
	shapeA, shapeB shapes.Shape

	correct float64
	err     bool
}{
	{gutils.Range[float64](0, 3), gutils.Range[float64](0, 3), shapes.Shape{3}, shapes.Shape{3}, float64(5), false},
	{gutils.Range[float64](0, 3), gutils.Range[float64](0, 3), shapes.Shape{3, 1}, shapes.Shape{3}, float64(5), false},
	{gutils.Range[float64](0, 3), gutils.Range[float64](0, 3), shapes.Shape{1, 3}, shapes.Shape{3}, float64(5), false},
	{gutils.Range[float64](0, 3), gutils.Range[float64](0, 3), shapes.Shape{3, 1}, shapes.Shape{3, 1}, float64(5), false},
	{gutils.Range[float64](0, 3), gutils.Range[float64](0, 3), shapes.Shape{1, 3}, shapes.Shape{3, 1}, float64(5), false},
	{gutils.Range[float64](0, 3), gutils.Range[float64](0, 3), shapes.Shape{3, 1}, shapes.Shape{1, 3}, float64(5), false},
	{gutils.Range[float64](0, 3), gutils.Range[float64](0, 3), shapes.Shape{1, 3}, shapes.Shape{1, 3}, float64(5), false},

	// // differing size
	{gutils.Range[float64](0, 4), gutils.Range[float64](0, 3), shapes.Shape{4}, shapes.Shape{3}, 0, true},

	// // A is not a matrix
	{gutils.Range[float64](0, 4), gutils.Range[float64](0, 3), shapes.Shape{2, 2}, shapes.Shape{3}, 0, true},
}

var innerTestsFloat32 = []struct {
	a, b           []float32
	shapeA, shapeB shapes.Shape

	correct float32
	err     bool
}{
	{gutils.Range[float32](0, 3), gutils.Range[float32](0, 3), shapes.Shape{3}, shapes.Shape{3}, float32(5), false},
	{gutils.Range[float32](0, 3), gutils.Range[float32](0, 3), shapes.Shape{3, 1}, shapes.Shape{3}, float32(5), false},
	{gutils.Range[float32](0, 3), gutils.Range[float32](0, 3), shapes.Shape{1, 3}, shapes.Shape{3}, float32(5), false},
	{gutils.Range[float32](0, 3), gutils.Range[float32](0, 3), shapes.Shape{3, 1}, shapes.Shape{3, 1}, float32(5), false},
	{gutils.Range[float32](0, 3), gutils.Range[float32](0, 3), shapes.Shape{1, 3}, shapes.Shape{3, 1}, float32(5), false},
	{gutils.Range[float32](0, 3), gutils.Range[float32](0, 3), shapes.Shape{3, 1}, shapes.Shape{1, 3}, float32(5), false},
	{gutils.Range[float32](0, 3), gutils.Range[float32](0, 3), shapes.Shape{1, 3}, shapes.Shape{1, 3}, float32(5), false},

	// // differing size
	{gutils.Range[float32](0, 4), gutils.Range[float32](0, 3), shapes.Shape{4}, shapes.Shape{3}, 0, true},

	// // A is not a matrix
	{gutils.Range[float32](0, 4), gutils.Range[float32](0, 3), shapes.Shape{2, 2}, shapes.Shape{3}, 0, true},
}

func TestDense_Inner(t *testing.T) {
	for i, its := range innerTestsFloat64 {
		a := New[float64](WithShape(its.shapeA...), WithBacking(its.a))
		b := New[float64](WithShape(its.shapeB...), WithBacking(its.b))

		T, err := a.Inner(b)
		if checkErr(t, its.err, err, "Inner", i) {
			continue
		}
		assert.Equal(t, its.correct, T)
	}

	for i, its := range innerTestsFloat32 {
		a := New[float32](WithShape(its.shapeA...), WithBacking(its.a))
		b := New[float32](WithShape(its.shapeB...), WithBacking(its.b))

		T, err := a.Inner(b)
		if checkErr(t, its.err, err, "Inner", i) {
			continue
		}
		assert.Equal(t, its.correct, T)
	}
}

type linalgTest[DT any] struct {
	a, b           []DT
	shapeA, shapeB shapes.Shape
	transA, transB bool

	reuse, incr    []DT
	shapeR, shapeI shapes.Shape

	correct      []DT
	correctIncr  []DT
	correctShape shapes.Shape
	err          bool
	errIncr      bool
	errReuse     bool
}

var matVecMulTestsFloat64 = []linalgTest[float64]{
	// Float64s
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 3), // a, b
		shapes.Shape{2, 3}, shapes.Shape{3}, // shapeA, shapeB
		false, false, // transA, transB
		gutils.Range[float64](52, 54), gutils.Range[float64](100, 102), // reuse, incr
		shapes.Shape{2}, shapes.Shape{2}, // shapeR, shapeI
		[]float64{5, 14},     // correct
		[]float64{105, 115},  // correct Incr
		shapes.Shape{2},      // correctShape
		false, false, false}, // errors
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 3), shapes.Shape{2, 3}, shapes.Shape{3, 1}, false, false,
		gutils.Range[float64](52, 54), gutils.Range[float64](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float64{5, 14}, []float64{105, 115}, shapes.Shape{2}, false, false, false},
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 3), shapes.Shape{2, 3}, shapes.Shape{1, 3}, false, false,
		gutils.Range[float64](52, 54), gutils.Range[float64](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float64{5, 14}, []float64{105, 115}, shapes.Shape{2}, true, false, false}, // DEPRECATED v0.10.0 - this used to be true. This is no longer a valid operation

	// float64s with transposed matrix
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 2), shapes.Shape{2, 3}, shapes.Shape{2}, true, false,
		gutils.Range[float64](52, 55), gutils.Range[float64](100, 103), shapes.Shape{3}, shapes.Shape{3},
		[]float64{3, 4, 5}, []float64{103, 105, 107}, shapes.Shape{3}, false, false, false},

	// stupids : unpossible shapes (wrong A)
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 3), shapes.Shape{6}, shapes.Shape{3}, false, false,
		gutils.Range[float64](52, 54), gutils.Range[float64](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float64{5, 14}, []float64{105, 115}, shapes.Shape{2}, true, false, false},

	//stupids: bad A shape
	{gutils.Range[float64](0, 8), gutils.Range[float64](0, 3), shapes.Shape{4, 2}, shapes.Shape{3}, false, false,
		gutils.Range[float64](52, 54), gutils.Range[float64](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float64{5, 14}, []float64{105, 115}, shapes.Shape{2}, true, false, false},

	//stupids: bad B shape
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 6), shapes.Shape{2, 3}, shapes.Shape{3, 2}, false, false,
		gutils.Range[float64](52, 54), gutils.Range[float64](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float64{5, 14}, []float64{105, 115}, shapes.Shape{2}, true, false, false},

	//stupids: bad reuse (too small) - (if reuse is larger, its OK - a memory flag stating that it's overprovisioned will be added)
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 3), shapes.Shape{2, 3}, shapes.Shape{3}, false, false,
		gutils.Range[float64](52, 53), gutils.Range[float64](100, 102), shapes.Shape{1}, shapes.Shape{2},
		[]float64{5, 14}, []float64{105, 115}, shapes.Shape{2}, false, false, true},

	//stupids: bad incr shape (too small) - (if incr is larger, it's OK)
	{gutils.Range[float64](0, 6), gutils.Range[float64](0, 3), shapes.Shape{2, 3}, shapes.Shape{3}, false, false,
		gutils.Range[float64](52, 54), gutils.Range[float64](100, 101), shapes.Shape{2}, shapes.Shape{1},
		[]float64{5, 14}, []float64{105, 115}, shapes.Shape{2}, false, true, false},
}

var matVecMulTestsFloat32 = []linalgTest[float32]{
	// Float32s
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 3), // a, b
		shapes.Shape{2, 3}, shapes.Shape{3}, // shapeA, shapeB
		false, false, // transA, transB
		gutils.Range[float32](52, 54), gutils.Range[float32](100, 102), // reuse, incr
		shapes.Shape{2}, shapes.Shape{2}, // shapeR, shapeI
		[]float32{5, 14},     // correct
		[]float32{105, 115},  // correct Incr
		shapes.Shape{2},      // correctShape
		false, false, false}, // errors
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 3), shapes.Shape{2, 3}, shapes.Shape{3, 1}, false, false,
		gutils.Range[float32](52, 54), gutils.Range[float32](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float32{5, 14}, []float32{105, 115}, shapes.Shape{2}, false, false, false},
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 3), shapes.Shape{2, 3}, shapes.Shape{1, 3}, false, false,
		gutils.Range[float32](52, 54), gutils.Range[float32](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float32{5, 14}, []float32{105, 115}, shapes.Shape{2}, true, false, false}, // DEPRECATED v0.10.0 - this used to be true. This is no longer a valid operation

	// float32s with transposed matrix
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 2), shapes.Shape{2, 3}, shapes.Shape{2}, true, false,
		gutils.Range[float32](52, 55), gutils.Range[float32](100, 103), shapes.Shape{3}, shapes.Shape{3},
		[]float32{3, 4, 5}, []float32{103, 105, 107}, shapes.Shape{3}, false, false, false},

	// stupids : unpossible shapes (wrong A)
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 3), shapes.Shape{6}, shapes.Shape{3}, false, false,
		gutils.Range[float32](52, 54), gutils.Range[float32](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float32{5, 14}, []float32{105, 115}, shapes.Shape{2}, true, false, false},

	//stupids: bad A shape
	{gutils.Range[float32](0, 8), gutils.Range[float32](0, 3), shapes.Shape{4, 2}, shapes.Shape{3}, false, false,
		gutils.Range[float32](52, 54), gutils.Range[float32](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float32{5, 14}, []float32{105, 115}, shapes.Shape{2}, true, false, false},

	//stupids: bad B shape
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 6), shapes.Shape{2, 3}, shapes.Shape{3, 2}, false, false,
		gutils.Range[float32](52, 54), gutils.Range[float32](100, 102), shapes.Shape{2}, shapes.Shape{2},
		[]float32{5, 14}, []float32{105, 115}, shapes.Shape{2}, true, false, false},

	//stupids: bad reuse (too small) - (if reuse is larger, its OK - a memory flag stating that it's overprovisioned will be added)
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 3), shapes.Shape{2, 3}, shapes.Shape{3}, false, false,
		gutils.Range[float32](52, 53), gutils.Range[float32](100, 102), shapes.Shape{1}, shapes.Shape{2},
		[]float32{5, 14}, []float32{105, 115}, shapes.Shape{2}, false, false, true},

	//stupids: bad incr shape (too small) - (if incr is larger, it's OK)
	{gutils.Range[float32](0, 6), gutils.Range[float32](0, 3), shapes.Shape{2, 3}, shapes.Shape{3}, false, false,
		gutils.Range[float32](52, 54), gutils.Range[float32](100, 101), shapes.Shape{2}, shapes.Shape{1},
		[]float32{5, 14}, []float32{105, 115}, shapes.Shape{2}, false, true, false},
}

func matVecMulHelper[DT interface{ float64 | float32 }](t *testing.T, assert *assert.Assertions, matVecMulTests []linalgTest[DT]) {
	t.Helper()
	for i, mvmt := range matVecMulTests {
		a := New[DT](WithBacking(mvmt.a), WithShape(mvmt.shapeA...))
		b := New[DT](WithBacking(mvmt.b), WithShape(mvmt.shapeB...))

		if mvmt.transA {
			var err error
			if a, err = a.T(); err != nil {
				t.Error(err)
				continue
			}
		}
		T, err := a.MatVecMul(b)
		if checkErr(t, mvmt.err, err, "Safe", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()), "Case %d: Expected %v × %v = %v. Got %v instead", i, a.Shape(), b.Shape(), mvmt.correctShape, T.Shape())
		assert.True(T.DataOrder().IsRowMajor())
		assert.Equal(mvmt.correct, T.Data(), "Case %d. Incorect data", i)

		// incr
		incr := New[DT](WithBacking(mvmt.incr), WithShape(mvmt.shapeI...))
		T, err = a.MatVecMul(b, WithIncr(incr))
		if checkErr(t, mvmt.errIncr, err, "WithIncr", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsRowMajor())
		assert.Equal(mvmt.correctIncr, T.Data())

		// reuse
		reuse := New[DT](WithBacking(mvmt.reuse), WithShape(mvmt.shapeR...))
		T, err = a.MatVecMul(b, WithReuse(reuse))
		if checkErr(t, mvmt.errReuse, err, "WithReuse", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsRowMajor())
		assert.Equal(mvmt.correct, T.Data())

	}
}

func TestDense_MatVecMul(t *testing.T) {
	assert := assert.New(t)
	matVecMulHelper(t, assert, matVecMulTestsFloat64)
	matVecMulHelper(t, assert, matVecMulTestsFloat32)
}

var tensorMulTestCases = []struct {
	a, b           []float64
	shapeA, shapeB shapes.Shape
	axesA, axesB   []int

	correct      []float64
	correctShape shapes.Shape

	willErr bool
}{
	{a: gutils.Range[float64](0, 3*4*5), b: gutils.Range[float64](0, 2*3*4),
		shapeA: shapes.Shape{3, 4, 5}, shapeB: shapes.Shape{4, 3, 2},
		axesA: []int{1, 0}, axesB: []int{0, 1},
		correct:      []float64{4400, 4730, 4532, 4874, 4664, 5018, 4796, 5162, 4928, 5306},
		correctShape: shapes.Shape{5, 2}},
}

func TestDense_TensorMul(t *testing.T) {
	assert := assert.New(t)
	for i, tmt := range tensorMulTestCases {
		t.Run(fmt.Sprintf("test_%d", i), func(t *testing.T) {
			A := New[float64](WithBacking(tmt.a), WithShape(tmt.shapeA...))
			B := New[float64](WithBacking(tmt.b), WithShape(tmt.shapeB...))

			C, err := A.TensorMul(B, tmt.axesA, tmt.axesB)
			switch {
			case tmt.willErr:
				assert.NotNil(err, "test %d", i)
				return
			default:
				if !assert.Nil(err, "test %d", i) {
					return
				}
				assert.Equal(tmt.correct, C.Data(), "Test %d", i)
				assert.Equal(tmt.correctShape, C.Shape(), "Test %d", i)
			}
		})
	}
}
