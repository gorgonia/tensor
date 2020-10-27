package tensor

import (
	"math/rand"
	"testing"
	"testing/quick"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestDense_ShallowClone(t *testing.T) {
	T := New(Of(Float64), WithBacking([]float64{1, 2, 3, 4}))
	T2 := T.ShallowClone()
	T2.slice(0, 2)
	T2.Float64s()[0] = 1000

	assert.Equal(t, T.Data().([]float64)[0:2], T2.Data())
	assert.Equal(t, T.Engine(), T2.Engine())
	assert.Equal(t, T.oe, T2.oe)
	assert.Equal(t, T.flag, T2.flag)
}

func TestDense_Clone(t *testing.T) {
	assert := assert.New(t)
	cloneChk := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		if !q.Shape().Eq(a.Shape()) {
			t.Errorf("Shape Difference: %v %v", q.Shape(), a.Shape())
			return false
		}
		if len(q.Strides()) != len(a.Strides()) {
			t.Errorf("Stride Difference: %v %v", q.Strides(), a.Strides())
			return false
		}
		for i, s := range q.Strides() {
			if a.Strides()[i] != s {
				t.Errorf("Stride Difference: %v %v", q.Strides(), a.Strides())
				return false
			}
		}
		if q.o != a.o {
			t.Errorf("Data Order difference : %v %v", q.o, a.o)
			return false
		}

		if q.Δ != a.Δ {
			t.Errorf("Triangle Difference: %v  %v", q.Δ, a.Δ)
			return false
		}
		if q.flag != a.flag {
			t.Errorf("Flag difference : %v %v", q.flag, a.flag)
			return false
		}
		if q.e != a.e {
			t.Errorf("Engine difference; %T %T", q.e, a.e)
			return false
		}
		if q.oe != a.oe {
			t.Errorf("Optimized Engine difference; %T %T", q.oe, a.oe)
			return false
		}

		if len(q.transposeWith) != len(a.transposeWith) {
			t.Errorf("TransposeWith difference: %v %v", q.transposeWith, a.transposeWith)
			return false
		}

		assert.Equal(q.mask, a.mask, "mask difference")
		assert.Equal(q.maskIsSoft, a.maskIsSoft, "mask is soft ")
		return true
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(cloneChk, &quick.Config{Rand: r}); err != nil {
		t.Error(err)
	}
}

func TestDenseMasked(t *testing.T) {
	T := New(Of(Float64), WithShape(3, 2))
	T.ResetMask()
	assert.Equal(t, []bool{false, false, false, false, false, false}, T.mask)

}

func TestFromScalar(t *testing.T) {
	T := New(FromScalar(3.14))
	data := T.Float64s()
	assert.Equal(t, []float64{3.14}, data)
}

func Test_recycledDense(t *testing.T) {
	T := recycledDense(Float64, ScalarShape())
	assert.Equal(t, float64(0), T.Data())
	assert.Equal(t, StdEng{}, T.e)
	assert.Equal(t, StdEng{}, T.oe)
}

func TestDense_unsqueeze(t *testing.T) {
	assert := assert.New(t)
	T := New(WithShape(3, 3, 2), WithBacking([]float64{
		1, 2, 3, 4, 5, 6,
		60, 50, 40, 30, 20, 10,
		100, 200, 300, 400, 500, 600,
	}))

	if err := T.unsqueeze(0); err != nil {
		t.Fatal(err)
	}

	assert.True(T.Shape().Eq(Shape{1, 3, 3, 2}))
	assert.Equal([]int{6, 6, 2, 1}, T.Strides()) // if you do shapes.CalcStrides() it'd be {18,6,2,1}

	// reset
	T.Reshape(3, 3, 2)

	if err := T.unsqueeze(1); err != nil {
		t.Fatal(err)
	}
	assert.True(T.Shape().Eq(Shape{3, 1, 3, 2}))
	assert.Equal([]int{6, 2, 2, 1}, T.Strides())

	// reset
	T.Reshape(3, 3, 2)
	if err := T.unsqueeze(2); err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", T)
	assert.True(T.Shape().Eq(Shape{3, 3, 1, 2}))
	assert.Equal([]int{6, 2, 1, 1}, T.Strides())

	// reset
	T.Reshape(3, 3, 2)
	if err := T.unsqueeze(3); err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", T)
	assert.True(T.Shape().Eq(Shape{3, 3, 2, 1}))
	assert.Equal([]int{6, 2, 1, 1}, T.Strides())
}
