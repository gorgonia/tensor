package storage

import (
	"reflect"
	"unsafe"

	"gorgonia.org/dtype"
)

type Array[T any] struct {
	Data  []T
	Bytes []byte
}

func bytesFromSlice[T any](dt dtype.Dtype, data []T) []byte {
	bytesHdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&data[0])),
		Len:  len(data) * int(dt.Size()),
		Cap:  cap(data) * int(dt.Size()),
	}
	raw := *(*[]byte)(unsafe.Pointer((&bytesHdr)))
	return raw
}
