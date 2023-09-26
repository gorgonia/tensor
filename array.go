package tensor

// type array[T any] struct {
// 	data []T         // the original data
// 	raw  []byte      // the original data but as a slice of bytes
// 	t    dtype.Dtype // the type, for run-time manipulation
// }

// func newArray[T any](data []T) array[T] {
// 	dt := dtype.Dtype{reflect.TypeOf(data).Elem()}

// 	bytesHdr := reflect.SliceHeader{
// 		Data: uintptr(unsafe.Pointer(&data[0])),
// 		Len:  len(data) * int(dt.Size()),
// 		Cap:  cap(data) * int(dt.Size()),
// 	}
// 	raw := *(*[]byte)(unsafe.Pointer((&bytesHdr)))
// 	return array[T]{
// 		data: data,
// 		raw:  raw,
// 		t:    dt,
// 	}
// }

// func (a array[T]) Data() []T { return a.data }
