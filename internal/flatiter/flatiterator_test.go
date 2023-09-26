package flatiter

import (
	"testing"

	"github.com/chewxy/inigo/values/tensor/internal"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
)

// newAP is  a helper function now
func newAP(shape Shape, strides []int) *AP {
	ap := internal.MakeAP(shape, strides, 0, 0)
	return &ap
}

var flatIterTests1 = []struct {
	shape   Shape
	strides []int

	correct []int
}{
	{shapes.ScalarShape(), []int{}, []int{0}},           // scalar
	{Shape{5}, []int{1}, []int{0, 1, 2, 3, 4}},          // vector
	{Shape{5, 1}, []int{1, 1}, []int{0, 1, 2, 3, 4}},    // colvec
	{Shape{1, 5}, []int{5, 1}, []int{0, 1, 2, 3, 4}},    // rowvec
	{Shape{2, 3}, []int{3, 1}, []int{0, 1, 2, 3, 4, 5}}, // basic mat
	{Shape{3, 2}, []int{1, 3}, []int{0, 3, 1, 4, 2, 5}}, // basic mat, transposed
	{Shape{2}, []int{2}, []int{0, 2}},                   // basic 2x2 mat, sliced: Mat[:, 1]
	{Shape{2, 2}, []int{5, 1}, []int{0, 1, 5, 6}},       // basic 5x5, sliced: Mat[1:3, 2,4]
	{Shape{2, 2}, []int{1, 5}, []int{0, 5, 1, 6}},       // basic 5x5, sliced: Mat[1:3, 2,4] then transposed

	{Shape{2, 3, 4}, []int{12, 4, 1}, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}}, // basic 3-Tensor
	{Shape{2, 4, 3}, []int{12, 1, 4}, []int{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}}, // basic 3-Tensor (under (0, 2, 1) transpose)
	{Shape{4, 2, 3}, []int{1, 12, 4}, []int{0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}}, // basic 3-Tensor (under (2, 0, 1) transpose)
	{Shape{3, 2, 4}, []int{4, 12, 1}, []int{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}}, // basic 3-Tensor (under (1, 0, 2) transpose)
	{Shape{4, 3, 2}, []int{1, 4, 12}, []int{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}}, // basic 3-Tensor (under (2, 1, 0) transpose)

	// ARTIFICIAL CASES - TODO
	// These cases should be impossible to reach in normal operation
	// You would have to specially construct these
	// {Shape{1, 5}, []int{1}, []int{0, 1, 2, 3, 4}},       // rowvec - NEARLY IMPOSSIBLE CASE- TODO
}

var flatIterSlices = []struct {
	slices   []SliceRange
	corrects [][]int
}{
	{[]SliceRange{nil}, [][]int{{0}}},
	{[]SliceRange{SR(0, 3, 1), SR(0, 5, 2), SR(0, 6, -1)}, [][]int{{0, 1, 2}, {0, 2, 4}, {4, 3, 2, 1, 0}}},
}

func TestFlatIterator(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var err error
	var nexts []int

	// basic stuff
	for i, fit := range flatIterTests1 {
		nexts = nexts[:0]
		err = nil
		ap = newAP(fit.shape, fit.strides)
		it = newFlatIterator(ap)
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			nexts = append(nexts, next)
		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			t.Error(err)
		}
		assert.Equal(fit.correct, nexts, "Test %d", i)
	}
}

func TestFlatIteratorReverse(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var err error
	var nexts []int

	// basic stuff
	for i, fit := range flatIterTests1 {
		nexts = nexts[:0]
		err = nil
		ap = newAP(fit.shape, fit.strides)
		it = newFlatIterator(ap)
		it.SetReverse()
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			nexts = append(nexts, next)
		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			t.Error(err)
		}
		// reverse slice
		for i, j := 0, len(nexts)-1; i < j; i, j = i+1, j-1 {
			nexts[i], nexts[j] = nexts[j], nexts[i]
		}
		// and then check
		assert.Equal(fit.correct, nexts, "Test %d", i)
	}
}
