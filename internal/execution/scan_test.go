package execution

import "testing"

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

func TestScanDefault(t *testing.T) {
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
	ScanDefault(a, retVal, 2, 2, 12, 12, 12, add)
	t.Logf("%v", retVal)

	// axis 1
	// 	dim0 = 2
	//	dimSize = 3
	//	outerStride = 12
	//	stride = 4
	// 	expected = irrelevant
	ScanDefault(a, retVal, 2, 3, 12, 4, 12, add)
	t.Logf("%v", retVal)

}
