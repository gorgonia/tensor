package array

import (
	"reflect"
	"unsafe"
)

// An Array holds the slice of the data type.
type Array[DT any] struct {
	data  []DT
	bytes []byte // the original data, but as a slice of bytes.
}

// Make makes an array
func Make[DT any](data []DT) Array[DT] {
	bs := BytesFromSlice[DT](data)
	return Array[DT]{
		data:  data,
		bytes: bs,
	}
}

func (a *Array[DT]) Data() []DT          { return a.data }
func (a *Array[DT]) DataAsBytes() []byte { return a.bytes }
func (a *Array[DT]) DataSize() int       { return len(a.data) }
func (a *Array[DT]) ScalarValue() DT     { return a.data[0] }
func (a *Array[DT]) MemSize() uintptr    { return uintptr(len(a.bytes)) }
func (a *Array[DT]) Uintptr() uintptr    { return uintptr(unsafe.Pointer(&a.bytes[0])) }

// Restore restores the array size to the original allocated length.
func (a *Array[DT]) Restore() {
	if cap(a.data) > len(a.data) {
		a.data = a.data[:cap(a.data)]
	}
}

// ResizeTo is used to resize the underlying data array to a new totalsize, which must be smaller than the array length.
func (a *Array[DT]) ResizeTo(totalsize int) {
	a.data = a.data[:totalsize]
}

// At returns the value at the given index. This assumes that the array is natively accessible. The check must be done elsewhere.
func (a *Array[DT]) At(i int) DT { return a.data[i] }

// SetAt sets a value at the given index. This assumes that the array is natively accessible. The check must be done elsewhere.
func (a *Array[DT]) SetAt(v DT, at int) { a.data[at] = v }

// Memset sets the values of the array to the given value. This assumes that the array is natively accessible. The check must be done elsewhere.
func (a *Array[DT]) Memset(v DT) {
	for i := range a.data {
		a.data[i] = v
	}
}

func (a *Array[DT]) Memclr() {
	var z DT
	for i := range a.data {
		a.data[i] = z
	}
}

func (a *Array[DT]) Slice(start, end, bytesStart, bytesEnd int) {
	a.data = a.data[start:end]
	a.bytes = a.bytes[bytesStart:bytesEnd]
}

// Use is used to set the values of the array. This should not be used at all. This function is exposed only for testing.
func (a *Array[DT]) Use(data []DT) {
	a.data = data
	a.bytes = BytesFromSlice[DT](data)
}

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
