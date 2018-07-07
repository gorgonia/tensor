package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var colMajorTraceTests = []struct {
	data interface{}

	correct interface{}
	err     bool
}{
	{[]int{0, 1, 2, 3, 4, 5}, int(4), false},
	{[]int8{0, 1, 2, 3, 4, 5}, int8(4), false},
	{[]int16{0, 1, 2, 3, 4, 5}, int16(4), false},
	{[]int32{0, 1, 2, 3, 4, 5}, int32(4), false},
	{[]int64{0, 1, 2, 3, 4, 5}, int64(4), false},
	{[]uint{0, 1, 2, 3, 4, 5}, uint(4), false},
	{[]uint8{0, 1, 2, 3, 4, 5}, uint8(4), false},
	{[]uint16{0, 1, 2, 3, 4, 5}, uint16(4), false},
	{[]uint32{0, 1, 2, 3, 4, 5}, uint32(4), false},
	{[]uint64{0, 1, 2, 3, 4, 5}, uint64(4), false},
	{[]float32{0, 1, 2, 3, 4, 5}, float32(4), false},
	{[]float64{0, 1, 2, 3, 4, 5}, float64(4), false},
	{[]complex64{0, 1, 2, 3, 4, 5}, complex64(4), false},
	{[]complex128{0, 1, 2, 3, 4, 5}, complex128(4), false},
	{[]bool{true, false, true, false, true, false}, nil, true},
}

func TestColMajor_Dense_Trace(t *testing.T) {
	assert := assert.New(t)
	for i, tts := range colMajorTraceTests {
		T := New(WithShape(2, 3), AsFortran(tts.data))
		trace, err := T.Trace()

		if checkErr(t, tts.err, err, "Trace", i) {
			continue
		}
		assert.Equal(tts.correct, trace)

		//
		T = New(WithBacking(tts.data))
		_, err = T.Trace()
		if err == nil {
			t.Error("Expected an error when Trace() on non-matrices")
		}
	}
}

var colMajorInnerTests = []struct {
	a, b           interface{}
	shapeA, shapeB Shape

	correct interface{}
	err     bool
}{
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{3, 1}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{3, 1}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{1, 3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{1, 3}, float64(5), false},

	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{3, 1}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{3, 1}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{1, 3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{1, 3}, float32(5), false},

	// stupids: type differences
	{Range(Int, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, nil, true},
	{Range(Float32, 0, 3), Range(Byte, 0, 3), Shape{3}, Shape{3}, nil, true},
	{Range(Float64, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3}, nil, true},
	{Range(Float32, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, nil, true},

	// differing size
	{Range(Float64, 0, 4), Range(Float64, 0, 3), Shape{4}, Shape{3}, nil, true},

	// A is not a matrix
	{Range(Float64, 0, 4), Range(Float64, 0, 3), Shape{2, 2}, Shape{3}, nil, true},
}

func TestColMajor_Dense_Inner(t *testing.T) {
	for i, its := range colMajorInnerTests {
		a := New(WithShape(its.shapeA...), AsFortran(its.a))
		b := New(WithShape(its.shapeB...), AsFortran(its.b))

		T, err := a.Inner(b)
		if checkErr(t, its.err, err, "Inner", i) {
			continue
		}

		assert.Equal(t, its.correct, T)
	}
}

var colMajorMatVecMulTests = []linalgTest{
	// Float64s
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, false},
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3, 1}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, false},
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{1, 3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, false},

	// float64s with transposed matrix
	{Range(Float64, 0, 6), Range(Float64, 0, 2), Shape{2, 3}, Shape{2}, true, false,
		Range(Float64, 52, 55), Range(Float64, 100, 103), Shape{3}, Shape{3},
		[]float64{3, 4, 5}, []float64{103, 105, 107}, []float64{106, 109, 112}, Shape{3}, false, false, false},

	// Float32s
	{Range(Float32, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float32, 52, 54), Range(Float32, 100, 102), Shape{2}, Shape{2},
		[]float32{5, 14}, []float32{105, 115}, []float32{110, 129}, Shape{2}, false, false, false},
	{Range(Float32, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3, 1}, false, false,
		Range(Float32, 52, 54), Range(Float32, 100, 102), Shape{2}, Shape{2},
		[]float32{5, 14}, []float32{105, 115}, []float32{110, 129}, Shape{2}, false, false, false},
	{Range(Float32, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{1, 3}, false, false,
		Range(Float32, 52, 54), Range(Float32, 100, 102), Shape{2}, Shape{2},
		[]float32{5, 14}, []float32{105, 115}, []float32{110, 129}, Shape{2}, false, false, false},

	// stupids : unpossible shapes (wrong A)
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{6}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	//stupids: bad A shape
	{Range(Float64, 0, 8), Range(Float64, 0, 3), Shape{4, 2}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	//stupids: bad B shape
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	//stupids: bad reuse
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 55), Range(Float64, 100, 102), Shape{3}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, true},

	//stupids: bad incr shape
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 105), Shape{2}, Shape{5},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, true, false},

	// stupids: type mismatch A and B
	{Range(Float64, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B
	{Range(Float32, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B
	{Range(Float64, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B
	{Range(Float32, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B (non-Float)
	{Range(Float64, 0, 6), Range(Int, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch, reuse
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float32, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, true},

	// stupids: type mismatch, incr
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), Range(Float32, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, true, false},

	// stupids: type mismatch, incr not a Number
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3}, false, false,
		Range(Float64, 52, 54), []bool{true, true, true}, Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, true, false},
}

func TestColMajor_Dense_MatVecMul(t *testing.T) {
	assert := assert.New(t)
	for i, mvmt := range colMajorMatVecMulTests {
		a := New(WithShape(mvmt.shapeA...), AsFortran(mvmt.a))
		b := New(WithShape(mvmt.shapeB...), AsFortran(mvmt.b))

		if mvmt.transA {
			if err := a.T(); err != nil {
				t.Error(err)
				continue
			}
		}

		T, err := a.MatVecMul(b)
		if checkErr(t, mvmt.err, err, "Safe", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(mvmt.correct, T.Data())

		// incr
		incr := New(WithShape(mvmt.shapeI...), AsFortran(mvmt.incr))
		T, err = a.MatVecMul(b, WithIncr(incr))
		if checkErr(t, mvmt.errIncr, err, "WithIncr", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(mvmt.correctIncr, T.Data())

		// reuse
		reuse := New(WithShape(mvmt.shapeR...), AsFortran(mvmt.reuse))
		T, err = a.MatVecMul(b, WithReuse(reuse))
		if checkErr(t, mvmt.errReuse, err, "WithReuse", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(mvmt.correct, T.Data())

		// reuse AND incr
		T, err = a.MatVecMul(b, WithIncr(incr), WithReuse(reuse))
		if checkErr(t, mvmt.err, err, "WithReuse and WithIncr", i) {
			continue
		}
		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(mvmt.correctIncrReuse, T.Data())
	}
}

var colMajorMatMulTests = []linalgTest{
	// Float64s
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, false, false, false},

	// Float32s
	{Range(Float32, 0, 6), Range(Float32, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float32, 52, 56), Range(Float32, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float32{10, 28, 13, 40}, []float32{110, 130, 114, 143}, []float32{120, 158, 127, 183}, Shape{2, 2}, false, false, false},

	// Edge cases - Row Vecs (Float64)
	{Range(Float64, 0, 2), Range(Float64, 0, 3), Shape{2, 1}, Shape{1, 3}, false, false,
		Range(Float64, 10, 16), Range(Float64, 100, 106), Shape{2, 3}, Shape{2, 3},
		[]float64{0, 0, 0, 1, 0, 2}, []float64{100, 103, 101, 105, 102, 107}, []float64{100, 103, 101, 106, 102, 109}, Shape{2, 3}, false, false, false},
	{Range(Float64, 0, 2), Range(Float64, 0, 6), Shape{1, 2}, Shape{2, 3}, false, false,
		Range(Float64, 10, 13), Range(Float64, 100, 103), Shape{1, 3}, Shape{1, 3},
		[]float64{3, 4, 5}, []float64{103, 105, 107}, []float64{106, 109, 112}, Shape{1, 3}, false, false, false},
	{Range(Float64, 0, 2), Range(Float64, 0, 2), Shape{1, 2}, Shape{2, 1}, false, false,
		Range(Float64, 0, 1), Range(Float64, 100, 101), Shape{1, 1}, Shape{1, 1},
		[]float64{1}, []float64{101}, []float64{102}, Shape{1, 1}, false, false, false},

	// Edge cases - Row Vecs (Float32)
	{Range(Float32, 0, 2), Range(Float32, 0, 3), Shape{2, 1}, Shape{1, 3}, false, false,
		Range(Float32, 10, 16), Range(Float32, 100, 106), Shape{2, 3}, Shape{2, 3},
		[]float32{0, 0, 0, 1, 0, 2}, []float32{100, 103, 101, 105, 102, 107}, []float32{100, 103, 101, 106, 102, 109}, Shape{2, 3}, false, false, false},
	{Range(Float32, 0, 2), Range(Float32, 0, 6), Shape{1, 2}, Shape{2, 3}, false, false,
		Range(Float32, 10, 13), Range(Float32, 100, 103), Shape{1, 3}, Shape{1, 3},
		[]float32{3, 4, 5}, []float32{103, 105, 107}, []float32{106, 109, 112}, Shape{1, 3}, false, false, false},
	{Range(Float32, 0, 2), Range(Float32, 0, 2), Shape{1, 2}, Shape{2, 1}, false, false,
		Range(Float32, 0, 1), Range(Float32, 100, 101), Shape{1, 1}, Shape{1, 1},
		[]float32{1}, []float32{101}, []float32{102}, Shape{1, 1}, false, false, false},

	// stupids - bad shape (not matrices):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{6}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, true, false, false},

	// stupids - bad shape (incompatible shapes):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{6, 1}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, true, false, false},

	// stupids - bad shape (bad reuse shape):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 57), Range(Float64, 100, 104), Shape{5}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, false, false, true},

	// stupids - bad shape (bad incr shape):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{4},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, false, true, false},

	// stupids - type mismatch (a,b)
	{Range(Float64, 0, 6), Range(Float32, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, true, false, false},

	// stupids - type mismatch (a,b)
	{Range(Float32, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, true, false, false},

	// stupids type mismatch (b not float)
	{Range(Float64, 0, 6), Range(Int, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, true, false, false},

	// stupids type mismatch (a not float)
	{Range(Int, 0, 6), Range(Int, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, true, false, false},

	// stupids: type mismatch (incr)
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float32, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, false, true, false},

	// stupids: type mismatch (reuse)
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float32, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 28, 13, 40}, []float64{110, 130, 114, 143}, []float64{120, 158, 127, 183}, Shape{2, 2}, false, false, true},

	// stupids: type mismatch (reuse)
	{Range(Float32, 0, 6), Range(Float32, 0, 6), Shape{2, 3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 56), Range(Float32, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float32{10, 28, 13, 40}, []float32{110, 130, 114, 143}, []float32{120, 158, 127, 183}, Shape{2, 2}, false, false, true},
}

func TestColMajorDense_MatMul(t *testing.T) {
	assert := assert.New(t)
	for i, mmt := range colMajorMatMulTests {
		a := New(WithShape(mmt.shapeA...), AsFortran(mmt.a))
		b := New(WithShape(mmt.shapeB...), AsFortran(mmt.b))

		T, err := a.MatMul(b)
		if checkErr(t, mmt.err, err, "Safe", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(mmt.correct, T.Data(), "Test %d", i)

		// incr
		incr := New(WithShape(mmt.shapeI...), AsFortran(mmt.incr))
		T, err = a.MatMul(b, WithIncr(incr))
		if checkErr(t, mmt.errIncr, err, "WithIncr", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correctIncr, T.Data())

		// reuse
		reuse := New(WithShape(mmt.shapeR...), AsFortran(mmt.reuse))
		T, err = a.MatMul(b, WithReuse(reuse))

		if checkErr(t, mmt.errReuse, err, "WithReuse", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correct, T.Data())

		// reuse AND incr
		T, err = a.MatMul(b, WithIncr(incr), WithReuse(reuse))
		if checkErr(t, mmt.err, err, "WithIncr and WithReuse", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correctIncrReuse, T.Data())
	}
}

var colMajorOuterTests = []linalgTest{
	// Float64s
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		false, false, false},

	// Float32s
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float32, 52, 61), Range(Float32, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float32{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float32{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float32{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		false, false, false},

	// stupids - a or b not vector
	{Range(Float64, 0, 3), Range(Float64, 0, 6), Shape{3}, Shape{3, 2}, false, false,
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		true, false, false},

	//	stupids - bad incr shape
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float64, 52, 61), Range(Float64, 100, 106), Shape{3, 3}, Shape{3, 2},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		false, true, false},

	// stupids - bad reuse shape
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float64, 52, 58), Range(Float64, 100, 109), Shape{3, 2}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		false, false, true},

	// stupids - b not Float
	{Range(Float64, 0, 3), Range(Int, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		true, false, false},

	// stupids - a not Float
	{Range(Int, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		true, false, false},

	// stupids - a-b type mismatch
	{Range(Float64, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		true, false, false},

	// stupids a-b type mismatch
	{Range(Float32, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, false, false,
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 103, 106, 101, 105, 109, 102, 107, 112}, []float64{100, 103, 106, 101, 106, 111, 102, 109, 116}, Shape{3, 3},
		true, false, false},
}

func TestColMajor_Dense_Outer(t *testing.T) {
	assert := assert.New(t)
	for i, ot := range colMajorOuterTests {
		a := New(WithShape(ot.shapeA...), AsFortran(ot.a))
		b := New(WithShape(ot.shapeB...), AsFortran(ot.b))

		T, err := a.Outer(b)
		if checkErr(t, ot.err, err, "Safe", i) {
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(ot.correct, T.Data())

		// incr
		incr := New(WithShape(ot.shapeI...), AsFortran(ot.incr))
		T, err = a.Outer(b, WithIncr(incr))
		if checkErr(t, ot.errIncr, err, "WithIncr", i) {
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(ot.correctIncr, T.Data())

		// reuse
		reuse := New(WithShape(ot.shapeR...), AsFortran(ot.reuse))
		T, err = a.Outer(b, WithReuse(reuse))
		if checkErr(t, ot.errReuse, err, "WithReuse", i) {
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(ot.correct, T.Data())

		// reuse AND incr
		T, err = a.Outer(b, WithIncr(incr), WithReuse(reuse))
		if err != nil {
			t.Errorf("Reuse and Incr error'd %+v", err)
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.True(T.DataOrder().IsColMajor())
		assert.Equal(ot.correctIncrReuse, T.Data())
	}
}
