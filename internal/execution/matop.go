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
