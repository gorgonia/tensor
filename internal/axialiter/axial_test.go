package axialiter

import (
	"testing"

	"gorgonia.org/tensor/internal"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

type NoOpError interface {
	error
	NoOp()
}

// newAP is  a helper function now
func newAP(shape Shape, strides []int) *AP {
	ap := internal.MakeAP(shape, strides, 0, 0)
	return &ap
}

var axialIterTests1 = []struct {
	shape   shapes.Shape
	strides []int
	axis    int
	axisSz  int
	fixed   bool

	correct []int
}{
	{
		shape:   shapes.Shape{2, 3, 4},
		strides: []int{12, 4, 1},
		axis:    0,
		axisSz:  2,
		fixed:   true,
		correct: []int{
			0, 1, 2, 3,
			4, 5, 6, 7,
			8, 9, 10, 11,
		},
	},
	{
		shape:   shapes.Shape{2, 3, 4},
		strides: []int{12, 4, 1},
		axis:    1,
		axisSz:  3,
		fixed:   true,
		correct: []int{0, 1, 2, 3, 12, 13, 14, 15},
	},
	{
		shape:   shapes.Shape{2, 3, 4},
		strides: []int{12, 4, 1},
		axis:    2,
		axisSz:  4,
		fixed:   true,
		correct: []int{0, 4, 8, 12, 16, 20},
	},
	{
		shape:   shapes.Shape{2, 3, 4},
		strides: []int{12, 4, 1},
		axis:    0,
		axisSz:  2,
		fixed:   false,
		correct: []int{
			0, 1, 2, 3,
			4, 5, 6, 7,
			8, 9, 10, 11,
		},
	},
	{
		shape:   shapes.Shape{2, 3, 4},
		strides: []int{12, 4, 1},
		axis:    1,
		axisSz:  3,
		fixed:   false,
		correct: []int{0, 1, 2, 3, 12, 13, 14, 15},
	},
}

func TestAxialIterator(t *testing.T) {
	assert := assert.New(t)

	for i, test := range axialIterTests1 {
		t.Run("", func(t *testing.T) {
			ap := newAP(test.shape, test.strides)
			it := New(ap, test.axis, test.axisSz, test.fixed)

			var nexts []int
			var next int
			var err error
			for next, err = it.Next(); err == nil; next, err = it.Next() {
				nexts = append(nexts, next)
			}

			if _, ok := err.(NoOpError); err != nil && !ok {
				t.Error(err)
			}

			assert.Equal(test.correct, nexts, "Test %d", i)
		})

	}

}
