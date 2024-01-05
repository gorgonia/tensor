package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	gutils "gorgonia.org/tensor/internal/utils"
)

func TestMax(t *testing.T) {
	assert := assert.New(t)

	t.Run("Matrix, No Axes Provided", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))

		expected := float64(6)
		retVal, err := Max(a)
		assert.Nil(err)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.ScalarShape()))
		assert.Equal(expected, retVal.ScalarValue())
	})

	t.Run("Matrix, Axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := New[float64](WithShape(3), WithBacking([]float64{4, 5, 6}))
		retVal, err := Max(a, Along(0))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})

	t.Run("Matrix, Axis 1", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := New[float64](WithShape(2), WithBacking([]float64{3, 6}))
		retVal, err := Max(a, Along(1))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})

	t.Run("3D Tensor, No Axes", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := float64(23)
		retVal, err := Max(a)
		assert.Nil(err)
		assert.Equal(expected, retVal.ScalarValue())
	})

	t.Run("3D Tensor, Axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := New[float64](WithShape(3, 4), WithBacking([]float64{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}))
		retVal, err := Max(a, Along(0))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})

	t.Run("3D Tensor, Axis 1", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := New[float64](WithShape(2, 4), WithBacking([]float64{8, 9, 10, 11, 20, 21, 22, 23}))
		retVal, err := Max(a, Along(1))
		assert.Nil(err)
		assert.True(retVal.Eq(expected), "Expected \n%v\nGot:\n%v", expected, retVal)
	})

	t.Run("3D Tensor, Axis 2", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := New[float64](WithShape(2, 3), WithBacking([]float64{3, 7, 11, 15, 19, 23}))
		retVal, err := Max(a, Along(2))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})

	t.Run("3D Tensor, Axes 2,0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := New[float64](WithShape(3), WithBacking([]float64{15, 19, 23}))
		retVal, err := Max(a, Along(2, 0))
		assert.Nil(err)
		assert.True(retVal.Eq(expected), "Expected %v.\nGot %v\n", expected, retVal)
	})

	t.Run("3D Tensor, Axes 2,1", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := New[float64](WithShape(2), WithBacking([]float64{11, 23}))
		retVal, err := Max(a, Along(2, 1))
		assert.Nil(err)
		assert.True(retVal.Eq(expected), "Expected %v.\nGot %v\n", expected, retVal)
	})

	t.Run("3D Tensor, Axes 2,1, with bad reuse", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		r := New[float64](WithShape(2))
		_, err := Max(a, Along(2, 1), WithReuse(r))
		assert.NotNil(err)
	})

	t.Run("3D Tensor, Axes 2,1, with good reuse", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		r := New[float64](WithShape(3, 4))
		expected := New[float64](WithShape(2), WithBacking([]float64{11, 23}))
		retVal, err := Max(a, Along(2, 1), WithReuse(r))
		assert.Nil(err)
		assert.True(retVal.Eq(expected), "Expected %v.\nGot %v\n", expected, retVal)
	})
}

func TestSum(t *testing.T) {
	assert := assert.New(t)

	t.Run("Vector", func(t *testing.T) {
		a := New[float64](WithShape(3), WithBacking([]float64{1, 2, 3}))
		expected := float64(6)
		retVal, err := Sum(a)
		assert.Nil(err)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.ScalarShape()))
		assert.Equal(expected, retVal.ScalarValue())
	})

	t.Run("Matrix, No Axes", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := float64(21)
		retVal, err := Sum(a)
		assert.Nil(err)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.ScalarShape()))
		assert.Equal(expected, retVal.ScalarValue())
	})

	t.Run("Matrix, Axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := New[float64](WithShape(3), WithBacking([]float64{5, 7, 9}))
		retVal, err := Sum(a, Along(0))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})

	t.Run("Matrix, Axis 1", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := New[float64](WithShape(2), WithBacking([]float64{6, 15}))
		retVal, err := Sum(a, Along(1))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})

	t.Run("3D Tensor, No Axes", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := float64(276)
		retVal, err := Sum(a)
		assert.Nil(err)
		assert.Equal(expected, retVal.ScalarValue())
	})

	t.Run("3D Tensor, Axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := New[float64](WithShape(3, 4), WithBacking([]float64{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34}))
		retVal, err := Sum(a, Along(0))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})
}

func TestProd(t *testing.T) {
	assert := assert.New(t)

	t.Run("Vector", func(t *testing.T) {
		a := New[float64](WithShape(3), WithBacking([]float64{1, 2, 3}))
		expected := float64(6)
		retVal, err := Prod(a)
		assert.Nil(err)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.ScalarShape()))
		assert.Equal(expected, retVal.ScalarValue())
	})

	t.Run("Matrix, No Axes", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := float64(720)
		retVal, err := Prod(a)
		assert.Nil(err)
		assert.NotNil(retVal)
		assert.True(retVal.Shape().Eq(shapes.ScalarShape()))
		assert.Equal(expected, retVal.ScalarValue())
	})

	t.Run("Matrix, Axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := New[float64](WithShape(3), WithBacking([]float64{4, 10, 18}))
		retVal, err := Prod(a, Along(0))
		assert.Nil(err)
		assert.True(retVal.Eq(expected))
	})

	t.Run("Matrix, Axis 1", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
		expected := New[float64](WithShape(2), WithBacking([]float64{6, 120}))
		retVal, err := Prod(a, Along(1))
		assert.Nil(err)
		assert.True(retVal.Eq(expected), "Expected\n%v\nGot\n%v", expected, retVal)
	})

	t.Run("3D Tensor, No Axes", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		expected := float64(0)
		retVal, err := Prod(a)
		assert.Nil(err)
		assert.Equal(expected, retVal.ScalarValue())
	})

}

func TestLt(t *testing.T) {
	a := New[int](WithShape(2, 3), WithBacking(gutils.Range[int](0, 6)))
	b := New[int](WithShape(2, 3), WithBacking(gutils.Range[int](3, 9)))

	expected := New[int](WithShape(2, 3), WithBacking([]int{1, 1, 1, 1, 1, 1}))
	ret, err := Lt(a, b)
	if err != nil {
		t.Fatal(err)
	}
	retTint := ret.(*Dense[int])
	if !retTint.Eq(expected) {
		t.Fatalf("Expected %v, got %v", expected, ret)
	}

	// As Bool:
	expectedBool := New[bool](WithShape(2, 3), WithBacking([]bool{true, true, true, true, true, true}))
	ret, err = Lt(a, b, As(dtype.Bool))
	if err != nil {
		t.Fatal(err)
	}
	retTbool := ret.(*Dense[bool])
	if !retTbool.Eq(expectedBool) {
		t.Fatalf("Expected %v, got %v", expected, ret)
	}
}

func TestDense_Argmax(t *testing.T) {
	assert := assert.New(t)

	t.Run("2D tensor, axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking(gutils.Range[float64](0, 6)))
		ret, err := a.Argmax(0)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(3), WithBacking([]int{1, 1, 1}))
		assert.Equal(expected, ret)
	})

	t.Run("3D tensor, axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		ret, err := a.Argmax(0)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(3, 4), WithBacking([]int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}))
		assert.Equal(expected, ret)
	})

	t.Run("3D tensor, axis 1", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		ret, err := a.Argmax(1)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(2, 4), WithBacking([]int{
			2, 2, 2, 2,
			2, 2, 2, 2}))
		assert.Equal(expected, ret)
	})

	t.Run("3D tensor, axis 2", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		ret, err := a.Argmax(2)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(2, 3), WithBacking([]int{
			3, 3, 3,
			3, 3, 3}))
		assert.Equal(expected, ret)
	})
}

func TestDense_Argmin(t *testing.T) {
	assert := assert.New(t)
	t.Run("2D tensor, axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3), WithBacking(gutils.Range[float64](0, 6)))
		ret, err := a.Argmin(0)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(3), WithBacking([]int{0, 0, 0}))
		assert.Equal(expected, ret)
	})

	t.Run("3D tensor, axis 0", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		ret, err := a.Argmin(0)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(3, 4), WithBacking([]int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}))
		assert.Equal(expected, ret)
	})

	t.Run("3D tensor, axis 1", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		ret, err := a.Argmin(1)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(2, 4), WithBacking([]int{
			0, 0, 0, 0,
			0, 0, 0, 0,
		}))
		assert.Equal(expected, ret)
	})

	t.Run("3D tensor, axis 2", func(t *testing.T) {
		a := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
		ret, err := a.Argmin(2)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		expected := New[int](WithShape(2, 3), WithBacking([]int{
			0, 0, 0,
			0, 0, 0}))
		assert.Equal(expected, ret)
	})
}
