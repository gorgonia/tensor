//go:build !nounsafe
// +build !nounsafe

package internal

import (
	"reflect"
	"unsafe"
)

func SliceFromMemory[DT any](m Memory) []DT {
	var z DT

	ptr := m.Uintptr()
	sz := m.MemSize()
	elements := sz / unsafe.Sizeof(z)
	hdr := reflect.SliceHeader{
		Data: ptr,
		Len:  int(elements),
		Cap:  int(elements),
	}
	return *(*[]DT)(unsafe.Pointer(&hdr))
}

func SliceAsMemory[DT any](data []DT) Memory {
	var z DT
	sz := unsafe.Sizeof(z)
	hdr := *(*reflect.SliceHeader)(unsafe.Pointer(&data))
	return &SliceMem{
		SliceHeader: hdr,
		sz:          uintptr(hdr.Cap) * sz,
	}
}

func CalcMemSize[DT any](size int) int64 {
	var z DT
	return int64(unsafe.Sizeof(z)) * int64(size)
}

// Overlaps checks if two slices overlaps one another or not
func Overlaps[DT any](a, b []DT) bool {
	ca, cb := cap(a), cap(b)
	if ca == 0 || cb == 0 {
		return false
	}
	fstA, fstB := &a[0], &b[0]
	if fstA == fstB {
		return true
	}
	var v DT
	sz := unsafe.Sizeof(v)
	aptr, bptr := uintptr(unsafe.Pointer(fstA)), uintptr(unsafe.Pointer(fstB))
	capA := aptr + uintptr(ca)*sz
	capB := bptr + uintptr(cb)*sz
	/*
		Overlap:
		+-----+-----+-----+-----+-----+-----+-----+-----+
		|  A  |     |  B  |     | cap |     |     | cap |
		| ptr |     | ptr |     |  A  |     |     |  B  |
		+-----+-----+-----+-----+-----+-----+-----+-----+
	*/
	switch {
	case aptr < bptr:
		if bptr < capA {
			return true
		}
	case aptr > bptr:
		if aptr < capB {
			return true
		}
	}
	return false
}

func SliceEqMeta[DTa, DTb any](a []DTa, b []DTb) bool {
	if len(a) != len(b) {
		return false
	}
	if len(a) == 0 {
		return true // b will also be 0 length, so it will be true
	}
	fstA, fstB := uintptr(unsafe.Pointer(&a[0])), uintptr(unsafe.Pointer(&b[0]))
	return fstA == fstB
}
