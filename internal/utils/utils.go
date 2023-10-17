package gutils

import (
	"reflect"
	"unsafe"

	"gorgonia.org/dtype"
)

func BytesFromSlice[T any](data []T) []byte {
	var v T
	sz := unsafe.Sizeof(v)
	bytesHdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&data[0])),
		Len:  len(data) * int(sz),
		Cap:  cap(data) * int(sz),
	}
	raw := *(*[]byte)(unsafe.Pointer((&bytesHdr)))
	return raw
}

func SliceFromBytes[T any](data []byte) []T {
	var v T
	sz := int(unsafe.Sizeof(v))
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(&data[0])),
		Len:  len(data) / sz,
		Cap:  len(data) / sz,
	}
	return *(*[]T)(unsafe.Pointer(&hdr))
}

func GetDatatype[T any]() dtype.Datatype[T] { return dtype.Datatype[T]{} }
