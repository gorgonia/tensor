package execution

// ReduceFirstN reduces the first axis of a given tensor with a function that is func(a, b []T).
func ReduceFirstN[T any](data, retVal []T, split, size int, fn func(a, b []T)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

// ReduceFirst reduces the first axis of a given tensor with a function that is func(a, b T) T.
func ReduceFirst[T any](data, retVal []T, split, size int, fn func(a, b T) T) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}

// ReduceFirstWithErr reduces the first axis of a given tensor with a function that is func(a, b T) (T, error).
func ReduceFirstWithErr[T any](data, retVal []T, split, size int, fn func(a, b T) (T, error)) (err error) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			if retVal[j], err = fn(retVal[j], data[j+start]); err != nil {
				return err
			}
		}
		start += split
	}
	return nil
}

// ReduceLastN reduces the last axis of a given tensor with a function that is func(a []T) T.
func ReduceLastN[T any](data, retVal []T, dimSize int, defaultValue T, fn func(data []T, defaultValue T) T) {
	var at int
	for start := 0; start <= len(data)-dimSize; start += dimSize {
		r := fn(data[start:start+dimSize], defaultValue)
		retVal[at] = r
		at++
	}
}

// ReduceLast reduces the last axis of a given tensor with a function that is func(a, b T) T.
func ReduceLast[T any](data, retVal []T, dimSize int, defaultValue T, fn func(T, T) T) {
	var at int

	for start := 0; start <= len(data)-dimSize; start += dimSize {
		r := Reduce(fn, defaultValue, data[start:start+dimSize])
		retVal[at] = r
		at++
	}
}

// ReduceLastWithErr reduces the last axis of a given tensor with a function that is func(a, b T) (T, error).
func ReduceLastWithErr[T any](data, retVal []T, dimSize int, defaultValue T, fn func(T, T) (T, error)) (err error) {
	var at int

	for start := 0; start <= len(data)-dimSize; start += dimSize {
		r, err := ReduceWithErr(fn, defaultValue, data[start:start+dimSize])
		if err != nil {
			return err
		}
		retVal[at] = r
		at++
	}
	return nil
}

// ReduceDefault reduces the an arbitrary axis of a given tensor with a function that is func(a, b T) T.
func ReduceDefault[T any](data, retVal []T, dim0, dimSize, outerStride, stride, expected int, fn func(T, T) T) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

// ReduceDefaultWithErr reduces the an arbitrary axis of a given tensor with a function that is func(a, b T) (T, error).
func ReduceDefaultWithErr[T any](data, retVal []T, dim0, dimSize, outerStride, stride, expected int, fn func(T, T) (T, error)) (err error) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				if retVal[writeTo], err = fn(retVal[writeTo], sliced[readFrom]); err != nil {
					return err
				}
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
	return nil
}

func Reduce[T any](fn func(T, T) T, defaultValue T, data []T) (retVal T) {
	retVal = defaultValue

	if len(data) == 0 {
		return retVal
	}
	for _, v := range data {
		retVal = fn(retVal, v)
	}
	return retVal
}

func ReduceWithErr[T any](fn func(T, T) (T, error), defaultValue T, data []T) (retVal T, err error) {
	retVal = defaultValue

	if len(data) == 0 {
		return retVal, nil
	}
	for _, v := range data {
		if retVal, err = fn(retVal, v); err != nil {
			return retVal, err
		}
	}
	return retVal, nil
}
