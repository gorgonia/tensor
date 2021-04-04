package storage

import (
	"github.com/stretchr/testify/assert"
	"reflect"
	"testing"
)

func TestCopy(t *testing.T) {
	// A longer than B
	a := headerFromSlice([]int{0, 1, 2, 3, 4})
	b := headerFromSlice([]int{10, 11})
	copied := Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 2, copied)
	assert.Equal(t, []int{10, 11, 2, 3, 4}, a.Ints())

	// B longer than A
	a = headerFromSlice([]int{10, 11})
	b = headerFromSlice([]int{0, 1, 2, 3, 4})
	copied = Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 2, copied)
	assert.Equal(t, []int{0, 1}, a.Ints())

	// A is empty
	a = headerFromSlice([]int{})
	b = headerFromSlice([]int{0, 1, 2, 3, 4})
	copied = Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 0, copied)

	// B is empty
	a = headerFromSlice([]int{0, 1, 2, 3, 4})
	b = headerFromSlice([]int{})
	copied = Copy(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 0, copied)
	assert.Equal(t, []int{0, 1, 2, 3, 4}, a.Ints())
}

func TestFill(t *testing.T) {
	// A longer than B
	a := headerFromSlice([]int{0, 1, 2, 3, 4})
	b := headerFromSlice([]int{10, 11})
	copied := Fill(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 5, copied)
	assert.Equal(t, []int{10, 11, 10, 11, 10}, a.Ints())

	// B longer than A
	a = headerFromSlice([]int{10, 11})
	b = headerFromSlice([]int{0, 1, 2, 3, 4})
	copied = Fill(reflect.TypeOf(1), &a, &b)

	assert.Equal(t, 2, copied)
	assert.Equal(t, []int{0, 1}, a.Ints())
}

func headerFromSlice(x interface{}) Header {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	xV := reflect.ValueOf(x)
	size := uintptr(xV.Len()) * xT.Elem().Size()
	return Header{
		Raw: FromMemory(xV.Pointer(), size),
	}
}
