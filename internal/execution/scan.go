package execution

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

func ScanFirst[T any](data, retVal []T, size, stride int, fn func(acc T, next T) T) {
	copy(retVal[0:stride], data[0:stride])
	// log.Printf("\ndata: %v\nretVal %v", data, retVal)
	for i := 1; i < size; i += stride {
		for j := 0; j < stride; j++ {
			acc := retVal[(i-1)*stride+j]
			retVal[i*stride+j] = fn(acc, data[i*stride+j])
		}
	}
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

// ScanDefault scans an arbitrary axis of a given tensor with a function that has the signature func(acc, next T) T.
// Note: despite mentioning "arbitrary axis" in the sentence prior to this, the arbitrariness is not guaranteed for first and last axes.
// Use `ScanFirst` and `ScanLast` for those
func ScanDefault[T any](data, retVal []T, dim0, dimSize, outerStride, stride, expected int, fn func(acc T, next T) T) {
	// log.Printf("\ndata: %v\nretVal %v", data, retVal)
	for i := 0; i < dim0*outerStride; i += outerStride {
		for j := 0; j < stride; j += expected {
			acc := data[i+j]
			retVal[i+j] = acc
			// log.Printf("acc = data[%d] = %v", i+j, acc)
			for k := stride; k < dimSize*stride; k += stride {
				readFrom := i + j + k

				// log.Printf("\tdata[%d + %d + %d = %d] = %v", i, j, k, readFrom, data[readFrom])
				acc = fn(acc, data[readFrom])
				retVal[readFrom] = acc
				// log.Printf("\tacc %v|%v", acc, retVal)
			}
		}
	}
}
