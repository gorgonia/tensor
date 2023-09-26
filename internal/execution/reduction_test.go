package execution

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReduce_All(t *testing.T) {
	assert := assert.New(t)
	// shape of (2,3,4)
	a := []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,

		13, 14, 15, 16,
		17, 18, 19, 20,
		21, 22, 23, 24,
	}

	add := func(a, b float64) float64 { return a + b }

	// reminder:
	// 	dim0 = shape[0]
	//	dimSize = shape[axis]
	//	outerStride = strides[0]
	// 	stride = strides[axis]
	// 	expected = retVal.Strides()[0]

	// axis 0
	retVal := make([]float64, 12)
	correct := []float64{14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}
	ReduceFirst(a, retVal, 12, 2, add)
	assert.Equal(correct, retVal)
	t.Logf("%v", retVal)

	// axis 1
	// 	dim0 = 2
	//	dimSize = 3
	// 	outerStride = 12
	//	stride = 4
	//	expected = 4
	retVal = make([]float64, 8)
	correct = []float64{15, 18, 21, 24, 51, 54, 57, 60}
	ReduceDefault(a, retVal, 2, 3, 12, 4, 4, add)
	assert.Equal(correct, retVal)
	t.Logf("%v", retVal)

	// axis 2
	retVal = make([]float64, 6)
	correct = []float64{10, 26, 42, 58, 74, 90}
	ReduceLast(a, retVal, 4, 0, add)
	assert.Equal(correct, retVal)
	t.Logf("%v", retVal)
}
