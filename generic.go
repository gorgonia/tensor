package tensor

import (
	"reflect"
	"unsafe"

	"golang.org/x/exp/constraints"
)

// cast casts a []T to a []U
func cast[T, U any](data []T) (retVal []U) {
	var t T
	tsz := unsafe.Sizeof(t)
	lenInBytes := tsz * uintptr(len(data))
	capInBytes := tsz * uintptr(cap(data))

	var u U
	usz := unsafe.Sizeof(u)
	l := lenInBytes / usz
	c := capInBytes / usz

	hdr := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	hdr.Cap = int(c)
	hdr.Len = int(l)
	return *(*[]U)(unsafe.Pointer(hdr))
}

// cast2 is like cast, but T and U are guaranteed to be the same size
func cast2[T, U any](data []T) (retVal []U) { return *(*[]U)(unsafe.Pointer(&data)) }

func min[T constraints.Ordered](a, b T) T {
	if a <= b {
		return a
	}
	return b
}

func max[T constraints.Ordered](a, b T) T {
	if a >= b {
		return a
	}
	return b
}
