package execution

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestScan(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	retVal := make([]float64, len(a))
	add := func(a, b float64) float64 { return a + b }
	Scan(add, a, retVal)
	t.Logf("%v", retVal)
}

func TestScanLastN(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	retVal := make([]float64, len(a))
	fn := func(data, retVal []float64) {
		retVal[0] = data[0]
		for i, v := range data[1:] {
			retVal[i+1] = retVal[i] + v
		}
	}
	ScanLastN(a, retVal, 2, fn)
	t.Logf("%v", retVal)
}

func TestScanLast(t *testing.T) {
	a := []float64{1, 2, 3, 4}
	retVal := make([]float64, len(a))
	add := func(a, b float64) float64 { return a + b }
	ScanLast(a, retVal, 2, add)
	t.Logf("%v", retVal)

}

func TestScan_All(t *testing.T) {
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
	retVal := make([]float64, len(a))
	add := func(a, b float64) float64 { return a + b }

	// reminder:
	// 	dim0 = shape[0]
	//	dimSize = shape[axis]
	//	outerStride = strides[0]
	// 	stride = strides[axis]
	// 	expected = retVal.Strides()[0]

	// axis 0
	// 	dim0 = 2
	//	dimSize = 2
	// 	outerStride = 12
	//	stride = 12
	//	ScanDefault(a, retVal, 2, 2, 12, 12, 1, add)
	correct := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36}
	ScanFirst(a, retVal, 2, 12, add)
	assert.Equal(correct, retVal)
	t.Logf("%v", retVal)

	// axis 1
	// 	dim0 = 2
	//	dimSize = 3
	//	outerStride = 12
	//	stride = 4
	// 	expected = irrelevant
	correct = []float64{1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 13, 14, 15, 16, 30, 32, 34, 36, 51, 54, 57, 60}
	ScanDefault(a, retVal, 2, 3, 12, 4, 1, add)
	assert.Equal(correct, retVal)
	t.Logf("%v", retVal)

	// axis 2
	correct = []float64{1, 3, 6, 10, 5, 11, 18, 26, 9, 19, 30, 42, 13, 27, 42, 58, 17, 35, 54, 74, 21, 43, 66, 90}
	ScanLast(a, retVal, 4, add)
	assert.Equal(correct, retVal)
	t.Logf("%v", retVal)
}
