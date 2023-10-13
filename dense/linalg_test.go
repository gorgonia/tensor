package dense

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	gutils "gorgonia.org/tensor/internal/utils"
)

// tests for SVD adapted from Gonum's SVD tests.
// Gonum's licence is listed at https://gonum.org/v1/gonum/license

type svdThinTest[DT float64 | float32] struct {
	data  []DT
	shape shapes.Shape

	correctSData []DT
	correctShape shapes.Shape

	correctUData  []DT
	correctUShape shapes.Shape

	correctVData  []DT
	correctVShape shapes.Shape
}

func svdtestsThin[DT float64 | float32]() []svdThinTest[DT] {
	return []svdThinTest[DT]{
		{
			[]DT{2, 4, 1, 3, 0, 0, 0, 0}, shapes.Shape{4, 2},
			[]DT{5.464985704219041, 0.365966190626258}, shapes.Shape{2},
			[]DT{-0.8174155604703632, -0.5760484367663209, -0.5760484367663209, 0.8174155604703633, 0, 0, 0, 0}, shapes.Shape{4, 2},
			[]DT{-0.4045535848337571, -0.9145142956773044, -0.9145142956773044, 0.4045535848337571}, shapes.Shape{2, 2},
		},

		{
			[]DT{1, 1, 0, 1, 0, 0, 0, 0, 0, 11, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 12, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 13, 3}, shapes.Shape{3, 11},
			[]DT{21.259500881097434, 1.5415021616856566, 1.2873979074613628}, shapes.Shape{3},
			[]DT{-0.5224167862273765, 0.7864430360363114, 0.3295270133658976, -0.5739526766688285, -0.03852203026050301, -0.8179818935216693, -0.6306021141833781, -0.6164603833618163, 0.4715056408282468}, shapes.Shape{3, 3},
			[]DT{
				-0.08123293141915189, 0.08528085505260324, -0.013165501690885152,
				-0.05423546426886932, 0.1102707844980355, 0.622210623111631,
				0, 0, 0,
				-0.0245733326078166, 0.510179651760153, 0.25596360803140994,
				0, 0, 0,
				0, 0, 0,
				-0.026997467150282436, -0.024989929445430496, -0.6353761248025164,
				0, 0, 0,
				-0.029662131661052707, -0.3999088672621176, 0.3662470150802212,
				-0.9798839760830571, 0.11328174160898856, -0.047702613241813366,
				-0.16755466189153964, -0.7395268089170608, 0.08395240366704032}, shapes.Shape{11, 3},
		},
	}
}

var svdtestsFull = []shapes.Shape{
	//{5, 5},
	//{5, 3},
	//{3, 5},
	//{150, 150},
	{200, 150},
	//{150, 200},
}

// calculate corrects
func calcSigma[DT float64 | float32](s, T *Dense[DT], shape shapes.Shape) (sigma *Dense[DT], err error) {
	sigma = New[DT](WithShape(shape...))
	for i := 0; i < internal.Min(shape[0], shape[1]); i++ {
		var idx int
		if idx, err = tensor.Ltoi(sigma.Shape(), sigma.Strides(), i, i); err != nil {
			return
		}
		sigma.Data()[idx] = s.Data()[i]
	}

	return
}

// test svd by doing the SVD, then calculating the corrects
func testSVD[DT float64 | float32](T, T2, s, u, v *Dense[DT], t string, i int) (err error) {
	var sigma, reconstructed *Dense[DT]

	if !allClose(T2.Data(), T.Data(), closeenough[DT]) {
		return errors.Errorf("A call to SVD modified the underlying data! %s Test %d", t, i)
	}

	shape := T2.Shape()
	if t == "thin" {
		shape = shapes.Shape{internal.Min(shape[0], shape[1]), internal.Min(shape[0], shape[1])}
	}

	if sigma, err = calcSigma(s, T, shape); err != nil {
		return
	}
	v2, err := v.T()
	if err != nil {
		return err
	}

	if reconstructed, err = u.MatMul(sigma, tensor.UseSafe); err != nil {
		return
	}

	if reconstructed, err = reconstructed.MatMul(v2, tensor.UseSafe); err != nil {
		return
	}

	if !allClose(T2.Data(), reconstructed.Data(), closeenough[DT]) {
		return errors.Errorf("Test %v - Expected reconstructed to be %v. Got %v instead", t, T2.Data(), reconstructed.Data())
	}
	return nil
}

func testDense_svd[DT float64 | float32](t *testing.T) {
	t.Helper()
	var T, T2, s, u, v *Dense[DT]
	var err error

	// gonum specific thin special cases
	for i, stts := range svdtestsThin[DT]() {
		T = New[DT](WithShape(stts.shape...), WithBacking(stts.data))
		T2 = T.Clone()

		if s, u, v, err = T.SVD(true, false); err != nil {
			t.Error(err)
			continue
		}

		if !allClose(T2.Data(), T.Data(), closeenough[DT]) {
			t.Errorf("A call to SVD modified the underlying data! Thin Test %d", i)
			continue
		}

		if !allClose(stts.correctSData, s.Data(), closeenough[DT]) {
			t.Errorf("Expected s = %v. Got %v instead", stts.correctSData, s.Data())
		}

		if !allClose(stts.correctUData, u.Data(), closeenough[DT]) {
			t.Errorf("Expected u = %v. Got %v instead", stts.correctUData, u.Data())
		}

		if !allClose(stts.correctVData, v.Data(), closeenough[DT]) {
			t.Errorf("Expected v = %v. Got %v instead", stts.correctVData, v.Data())
		}
	}
	// standard tests
	for i, stfs := range svdtestsFull {
		T = New[DT](WithShape(stfs...), WithBacking(gutils.Random[DT](stfs.TotalSize())))
		T2 = T.Clone()

		// full
		if s, u, v, err = T.SVD(true, true); err != nil {
			t.Error(err)
			fmt.Println(err)
			continue
		}
		if err = testSVD(T, T2, s, u, v, "full", i); err != nil {
			t.Error(err)
			fmt.Println(err)
			continue
		}
		// thin
		if s, u, v, err = T.SVD(true, false); err != nil {
			t.Error(err)
			continue
		}

		if err = testSVD(T, T2, s, u, v, "thin", i); err != nil {
			t.Error(err)
			continue
		}

		// none
		if s, u, v, err = T.SVD(false, false); err != nil {
			t.Error(err)
			continue
		}

		var svd mat.SVD
		var m *mat.Dense
		if m, err = ToMat64[DT](T); err != nil {
			t.Error(err)
			continue
		}

		if !svd.Factorize(m, mat.SVDFull) {
			t.Errorf("Unable to factorise %v", m)
			continue
		}

		if !allClose(s.Data(), convert[DT, float64](svd.Values(nil)), closeenough[DT]) {
			t.Errorf("Singular value mismatch between Full and None decomposition. Expected %v. Got %v instead", svd.Values(nil), s.Data())
		}

	}
	// this is illogical
	T = New[DT](WithShape(2, 2))
	if _, _, _, err = T.SVD(false, true); err == nil {
		t.Errorf("Expected an error!")
	}

	// if you do this, it is bad and you should feel bad
	T = New[DT](WithShape(2, 3, 4))
	if _, _, _, err = T.SVD(true, true); err == nil {
		t.Errorf("Expecetd an error: cannot SVD() a Tensor > 2 dimensions")
	}

	T = New[DT](WithShape(2))
	if _, _, _, err = T.SVD(true, true); err == nil {
		t.Errorf("Expecetd an error: cannot SVD() a Tensor < 2 dimensions")
	}
}

func TestDense_SVD(t *testing.T) {
	t.Run("float64", testDense_svd[float64])
	t.Run("float32", testDense_svd[float32])
}

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

type innerTest[DT float64 | float32] struct {
	a, b           []DT
	shapeA, shapeB shapes.Shape

	correct DT
	err     bool
}

func makeInnerTests[DT float64 | float32]() []innerTest[DT] {
	return []innerTest[DT]{
		{gutils.Range[DT](0, 3), gutils.Range[DT](0, 3), shapes.Shape{3}, shapes.Shape{3}, 5, false},
		{gutils.Range[DT](0, 3), gutils.Range[DT](0, 3), shapes.Shape{3, 1}, shapes.Shape{3}, 5, false},
		{gutils.Range[DT](0, 3), gutils.Range[DT](0, 3), shapes.Shape{1, 3}, shapes.Shape{3}, 5, false},
		{gutils.Range[DT](0, 3), gutils.Range[DT](0, 3), shapes.Shape{3, 1}, shapes.Shape{3, 1}, 5, false},
		{gutils.Range[DT](0, 3), gutils.Range[DT](0, 3), shapes.Shape{1, 3}, shapes.Shape{3, 1}, 5, false},
		{gutils.Range[DT](0, 3), gutils.Range[DT](0, 3), shapes.Shape{3, 1}, shapes.Shape{1, 3}, 5, false},
		{gutils.Range[DT](0, 3), gutils.Range[DT](0, 3), shapes.Shape{1, 3}, shapes.Shape{1, 3}, 5, false},

		// A and B are multidimensional. Note that `inner` doesn't really care about the shape!
		{gutils.Range[DT](0, 24), gutils.Range[DT](0, 24), shapes.Shape{2, 3, 4}, shapes.Shape{2, 3, 4}, 4324, false},

		// // differing size
		{gutils.Range[DT](0, 4), gutils.Range[DT](0, 3), shapes.Shape{4}, shapes.Shape{3}, 0, true},

		// // A is not a matrix
		{gutils.Range[DT](0, 4), gutils.Range[DT](0, 3), shapes.Shape{2, 2}, shapes.Shape{3}, 0, true},
	}
}

func TestDense_Inner(t *testing.T) {
	for i, its := range makeInnerTests[float64]() {
		a := New[float64](WithShape(its.shapeA...), WithBacking(its.a))
		b := New[float64](WithShape(its.shapeB...), WithBacking(its.b))

		T, err := a.Inner(b)
		if checkErr(t, its.err, err, "Inner (float64)", i) {
			continue
		}
		assert.Equal(t, its.correct, T)
	}

	for i, its := range makeInnerTests[float32]() {
		a := New[float32](WithShape(its.shapeA...), WithBacking(its.a))
		b := New[float32](WithShape(its.shapeB...), WithBacking(its.b))

		T, err := a.Inner(b)
		if checkErr(t, its.err, err, "Inner (float32)", i) {
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
