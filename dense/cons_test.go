package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

var onesTests = []struct {
	shape   shapes.Shape
	correct []int
}{
	{shapes.ScalarShape(), []int{1}},
	{shapes.Shape{2, 2}, []int{1, 1, 1, 1}},
}

func TestOnes(t *testing.T) {
	assert := assert.New(t)
	for _, ot := range onesTests {
		T := Ones[int](ot.shape...)
		assert.True(ot.shape.Eq(T.Shape()))
		assert.Equal(ot.correct, T.Data())
	}
}

// yes, it's a pun on eye tests, stop asking and go see your optometrist
var eyeTests = []struct {
	R, C, K int
	correct []int
}{
	{4, 4, 0, []int{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}},
	{4, 4, 1, []int{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
	{4, 4, 2, []int{0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
	{4, 4, 3, []int{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{4, 4, 4, []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{4, 4, -1, []int{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}},
	{4, 4, -2, []int{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}},
	{4, 4, -3, []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
	{4, 4, -4, []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	{4, 5, 0, []int{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}},
	{4, 5, 1, []int{0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1}},
	{4, 5, -1, []int{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0}},
}

func TestI(t *testing.T) {
	assert := assert.New(t)

	var T *Dense[int]
	for i, it := range eyeTests {
		T = I[int](it.R, it.C, it.K)
		assert.True(shapes.Shape{it.R, it.C}.Eq(T.Shape()))
		assert.Equal(it.correct, T.Data(), "Test %d-R: %d, C: %d K: %d", i, it.R, it.C, it.K)
	}

}
