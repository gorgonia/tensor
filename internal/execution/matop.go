package execution

func Transpose[T any](data []T, it Iterator) (out []T) {
	out = make([]T, len(data))
	var j int
	for i, err := it.Start(); err == nil; i, err = it.Next() {
		out[j] = data[i]
		j++
	}
	return out
}

func CopyIter[T any](dst, src []T, dit, sit Iterator) (err error) {
	var i, j int
	var validi, validj bool

loop:
	for {
		if i, validi, err = sit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break loop
		}
		if j, validj, err = dit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break loop
		}
		if validi && validj {
			dst[j] = src[i]
		}
	}
	return
}
