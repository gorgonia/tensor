package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShapeCalcStride(t *testing.T) {
	assert := assert.New(t)
	var s Shape

	// scalar shape
	s = Shape{}
	assert.Nil(CalcStrides(s))

	// vector shape
	s = Shape{1}
	assert.Equal([]int{1}, CalcStrides(s))

	s = Shape{2, 1}
	assert.Equal([]int{1, 1}, CalcStrides(s))

	s = Shape{1, 2}
	assert.Equal([]int{2, 1}, CalcStrides(s))

	s = Shape{2}
	assert.Equal([]int{1}, CalcStrides(s))

	// matrix strides
	s = Shape{2, 2}
	assert.Equal([]int{2, 1}, CalcStrides(s))

	s = Shape{5, 2}
	assert.Equal([]int{2, 1}, CalcStrides(s))

	// 3D strides
	s = Shape{2, 3, 4}
	assert.Equal([]int{12, 4, 1}, CalcStrides(s))

	// stupid shape
	s = Shape{-2, 1, 2}
	fail := func() {
		CalcStrides(s)
	}
	assert.Panics(fail)
}
