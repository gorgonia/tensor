package storage

import (
	"reflect"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestFill(t *testing.T) {
	// A longer than B
	a := headerFromSlice([]int{0, 1, 2, 3, 4})
	b := headerFromSlice([]int{10, 11})
	copied := Fill(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, copied, 5)
	assert.Equal(t, a.Ints(), []int{10, 11, 10, 11, 10})

	// B longer than A
	a = headerFromSlice([]int{10, 11})
	b = headerFromSlice([]int{0, 1, 2, 3, 4})
	copied = Fill(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, copied, 2)
	assert.Equal(t, a.Ints(), []int{0, 1})
}

func headerFromSlice(x interface{}) Header {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}

	xV := reflect.ValueOf(x)
	uptr := unsafe.Pointer(xV.Pointer())

	return Header{
		Ptr: uptr,
		L:   xV.Len(),
		C:   xV.Cap(),
	}
}
