package execution

import "log"

func Scan[T any](fn func(T, T) T, data []T, retVal []T) []T {
	if len(data) == 0 {
		return retVal
	}
	data = data[:]
	retVal = retVal[:len(data)]
	retVal[0] = data[0]
	for i, v := range data[1:] {
		retVal[i+1] = fn(retVal[i], v)
	}
	return retVal
}

func ScanLastN[T any](data, retVal []T, dimSize int, fn func(data, retVal []T)) {
	for start := 0; start <= len(data)-dimSize; start += dimSize {
		fn(data[start:start+dimSize], retVal[start:start+dimSize])
	}
}

func ScanLast[T any](data, retVal []T, dimSize int, fn func(acc T, next T) T) {
	for start := 0; start <= len(data)-dimSize; start += dimSize {
		Scan(fn, data[start:start+dimSize], retVal[start:start+dimSize])
	}
}

func ScanDefault[T any](data, retVal []T, dim0, dimSize, outerStride, stride, expected int, fn func(acc T, next T) T) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		slicedRetVal := retVal[start : start+outerStride]

		//var innerStart, strideTrack int
		//log.Printf("%v %v", slicedRetVal, sliced)
		for j := 0; j < dimSize; j++ {
			idx := j * stride

			for k := 0; k < stride; k++ {
				idx2 := idx + k
				log.Printf("j %d idx %d idx2 %v", j, idx, idx2)
				if j == 0 {
					slicedRetVal[idx2] = sliced[idx2]
					continue
				}
				idx1 := (j-1)*stride + k
				slicedRetVal[idx2] = fn(slicedRetVal[idx1], sliced[idx2])

			}
		}
	}
}
