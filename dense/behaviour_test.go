package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

func TestAPLConsBehaviour(t *testing.T) {
	assert := assert.New(t)
	a := []int{1, 2, 3, 4, 5}
	shape := shapes.Shape{2, 5}

	b := APLConsBehaviour(a, shape)
	assert.Equal([]int{1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, b)

	shape = shapes.Shape{7}
	b = APLConsBehaviour(a, shape)
	assert.Equal([]int{1, 2, 3, 4, 5, 1, 2}, b)
}
