package native

import (
	"reflect"
	"unsafe"

	. "gorgonia.org/tensor"
)

func Vector(t *Dense) (interface{}, error) {
	if err := checkNativeIterable(t, 1, t.Dtype()); err != nil {
		return nil, err
	}
	return t.Data(), nil
}

func Matrix(t *Dense) (interface{}, error) {
	if err := checkNativeIterable(t, 2, t.Dtype()); err != nil {
		return nil, err
	}

	shape := t.Shape()
	strides := t.Strides()
	typ := t.Dtype().Type
	rows := shape[0]
	cols := shape[1]
	rowStride := strides[0]

	retVal := reflect.MakeSlice(reflect.SliceOf(reflect.SliceOf(typ)), rows, rows)
	ptr := t.Uintptr()
	for i := 0; i < rows; i++ {
		e := retVal.Index(i)
		sh := (*reflect.SliceHeader)(unsafe.Pointer(e.Addr().Pointer()))
		sh.Data = uintptr(i*rowStride)*typ.Size() + ptr
		sh.Len = cols
		sh.Cap = cols
	}
	return retVal.Interface(), nil
}

func Tensor3(t *Dense) (interface{}, error) {
	if err := checkNativeIterable(t, 3, t.Dtype()); err != nil {
		return nil, err
	}
	shape := t.Shape()
	strides := t.Strides()
	typ := t.Dtype().Type

	layers := shape[0]
	rows := shape[1]
	cols := shape[2]
	layerStride := strides[0]
	rowStride := strides[1]
	retVal := reflect.MakeSlice(reflect.SliceOf(reflect.SliceOf(reflect.SliceOf(typ))), layers, layers)
	ptr := t.Uintptr()
	for i := 0; i < layers; i++ {
		el := retVal.Index(i)
		inner := reflect.MakeSlice(reflect.SliceOf(reflect.SliceOf(typ)), rows, rows)
		for j := 0; j < rows; j++ {
			e := inner.Index(j)
			sh := (*reflect.SliceHeader)(unsafe.Pointer(e.Addr().Pointer()))
			sh.Data = uintptr(i*layerStride+j*rowStride)*typ.Size() + ptr
			sh.Len = cols
			sh.Cap = cols
		}
		sh := (*reflect.SliceHeader)(unsafe.Pointer(el.Addr().Pointer()))
		sh.Data = inner.Index(0).Addr().Pointer()
		sh.Len = rows
		sh.Cap = rows
	}
	return retVal.Interface(), nil
}
