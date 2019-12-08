package tensor

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"
)

// array is the underlying generic array.
type array struct {
	storage.Header             // the header - the Go representation (a slice)
	t              Dtype       // the element type
	v              interface{} // an additional reference to the underlying slice. This is not strictly necessary, but does improve upon anything that calls .Data()
}

// makeHeader makes a array Header
func makeHeader(t Dtype, length int) storage.Header {
	return storage.Header{
		Ptr: malloc(t, length),
		L:   length,
		C:   length,
	}
}

// makeArray makes an array. The memory allocation is handled by Go
func makeArray(t Dtype, length int) array {
	hdr := makeHeader(t, length)
	return makeArrayFromHeader(hdr, t)
}

// makeArrayFromHeader makes an array given a header
func makeArrayFromHeader(hdr storage.Header, t Dtype) array {
	return array{
		Header: hdr,
		t:      t,
		v:      nil,
	}
}

// arrayFromSlice creates an array from a slice. If x is not a slice, it will panic.
func arrayFromSlice(x interface{}) array {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	elT := xT.Elem()

	xV := reflect.ValueOf(x)
	uptr := unsafe.Pointer(xV.Pointer())

	return array{
		Header: storage.Header{
			Ptr: uptr,
			L:   xV.Len(),
			C:   xV.Cap(),
		},
		t: Dtype{elT},
		v: x,
	}
}

// fromSlice populates the value from a slice
func (a *array) fromSlice(x interface{}) {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	elT := xT.Elem()
	xV := reflect.ValueOf(x)
	uptr := unsafe.Pointer(xV.Pointer())

	a.Ptr = uptr
	a.L = xV.Len()
	a.C = xV.Cap()
	a.t = Dtype{elT}
	a.v = x
}

// fromSliceOrTensor populates the value from a slice or anything that can form an array
func (a *array) fromSliceOrArrayer(x interface{}) {
	if T, ok := x.(arrayer); ok {
		xp := T.arrPtr()

		// if the underlying array hasn't been allocated, or not enough has been allocated
		if a.Ptr == nil || a.L < xp.L || a.C < xp.C {
			a.t = xp.t
			a.L = xp.L
			a.C = xp.C
			a.Ptr = malloc(a.t, a.L)
		}

		a.t = xp.t
		a.L = xp.L
		a.C = xp.C
		copyArray(a, T.arrPtr())
		a.v = nil    // tell the GC to release whatever a.v may hold
		a.forcefix() // fix it such that a.v has a value and is not nil
		return
	}
	a.fromSlice(x)
}

// fix fills the a.v empty interface{}  if it's not nil
func (a *array) fix() {
	if a.v == nil {
		a.forcefix()
	}
}

// forcefix fills the a.v empty interface{}. No checks are made if the thing is empty
func (a *array) forcefix() {
	sliceT := reflect.SliceOf(a.t.Type)
	ptr := unsafe.Pointer(&a.Header)
	val := reflect.Indirect(reflect.NewAt(sliceT, ptr))
	a.v = val.Interface()
}

// byteSlice casts the underlying slice into a byte slice. Useful for copying and zeroing, but not much else
func (a array) byteSlice() []byte {
	return storage.AsByteSlice(&a.Header, a.t.Type)
}

// sliceInto creates a slice. Instead of returning an array, which would cause a lot of reallocations, sliceInto expects a array to
// already have been created. This allows repetitive actions to be done without having to have many pointless allocation
func (a *array) sliceInto(i, j int, res *array) {
	c := a.C

	if i < 0 || j < i || j > c {
		panic(fmt.Sprintf("Cannot slice %v - index %d:%d is out of bounds", a, i, j))
	}

	res.L = j - i
	res.C = c - i

	if c-1 > 0 {
		res.Ptr = storage.ElementAt(i, a.Ptr, a.t.Size())
	} else {
		// don't advance pointer
		res.Ptr = a.Ptr
	}
	res.fix()
}

// slice slices an array
func (a array) slice(start, end int) array {
	if end > a.L {
		panic("Index out of range")
	}
	if end < start {
		panic("Index out of range")
	}

	L := end - start
	C := a.C - start

	var startptr unsafe.Pointer
	if a.C-start > 0 {
		startptr = storage.ElementAt(start, a.Ptr, a.t.Size())
	} else {
		startptr = a.Ptr
	}

	hdr := storage.Header{
		Ptr: startptr,
		L:   L,
		C:   C,
	}

	return makeArrayFromHeader(hdr, a.t)
}

// swap swaps the elements i and j in the array
func (a *array) swap(i, j int) {
	if a.t == String {
		ss := a.hdr().Strings()
		ss[i], ss[j] = ss[j], ss[i]
		return
	}
	if !isParameterizedKind(a.t.Kind()) {
		switch a.t.Size() {
		case 8:
			us := a.hdr().Uint64s()
			us[i], us[j] = us[j], us[i]
		case 4:
			us := a.hdr().Uint32s()
			us[i], us[j] = us[j], us[i]
		case 2:
			us := a.hdr().Uint16s()
			us[i], us[j] = us[j], us[i]
		case 1:
			us := a.hdr().Uint8s()
			us[i], us[j] = us[j], us[i]
		}
		return
	}

	size := int(a.t.Size())
	tmp := make([]byte, size)
	bs := a.byteSlice()
	is := i * size
	ie := is + size
	js := j * size
	je := js + size
	copy(tmp, bs[is:ie])
	copy(bs[is:ie], bs[js:je])
	copy(bs[js:je], tmp)
}

/* *Array is a Memory */

// Uintptr returns the pointer of the first value of the slab
func (a *array) Uintptr() uintptr { return uintptr(a.Ptr) }

// MemSize returns how big the slice is in bytes
func (a *array) MemSize() uintptr { return uintptr(a.L) * a.t.Size() }

// Pointer returns the pointer of the first value of the slab, as an unsafe.Pointer
func (a *array) Pointer() unsafe.Pointer { return a.Ptr }

// Data returns the representation of a slice.
func (a array) Data() interface{} {
	if a.v == nil {
		// build a type of []T
		shdr := reflect.SliceHeader{
			Data: uintptr(a.Header.Ptr),
			Len:  a.Header.L,
			Cap:  a.Header.C,
		}
		sliceT := reflect.SliceOf(a.t.Type)
		ptr := unsafe.Pointer(&shdr)
		val := reflect.Indirect(reflect.NewAt(sliceT, ptr))
		a.v = val.Interface()

	}
	return a.v
}

// Zero zeroes out the underlying array of the *Dense tensor.
func (a array) Zero() {
	if a.t.Kind() == reflect.String {
		ss := a.Strings()
		for i := range ss {
			ss[i] = ""
		}
		return
	}
	if !isParameterizedKind(a.t.Kind()) {
		ba := a.byteSlice()
		for i := range ba {
			ba[i] = 0
		}
		return
	}
	ptr := uintptr(a.Ptr)
	for i := 0; i < a.L; i++ {
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t.Type, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(reflect.Zero(a.t))
	}
}

func (a *array) hdr() *storage.Header { return &a.Header }
func (a *array) rtype() reflect.Type  { return a.t.Type }

/* MEMORY MOVEMENT STUFF */

// malloc is standard Go allocation of a block of memory - the plus side is that Go manages the memory
func malloc(t Dtype, length int) unsafe.Pointer {
	size := int(calcMemSize(t, length))
	s := make([]byte, size)
	return unsafe.Pointer(&s[0])
}

// calcMemSize calulates the memory size of an array (given its size)
func calcMemSize(dt Dtype, size int) int64 {
	return int64(dt.Size()) * int64(size)
}

// copyArray copies an array.
func copyArray(dst, src *array) int {
	if dst.t != src.t {
		panic("Cannot copy arrays of different types.")
	}
	return storage.Copy(dst.t.Type, &dst.Header, &src.Header)
}

func copyArraySliced(dst array, dstart, dend int, src array, sstart, send int) int {
	if dst.t != src.t {
		panic("Cannot copy arrays of different types.")
	}
	return storage.CopySliced(dst.t.Type, &dst.Header, dstart, dend, &src.Header, sstart, send)
}

// copyDense copies a DenseTensor
func copyDense(dst, src DenseTensor) int {
	if dst.Dtype() != src.Dtype() {
		panic("Cannot dopy DenseTensors of different types")
	}

	if ms, ok := src.(MaskedTensor); ok && ms.IsMasked() {
		if md, ok := dst.(MaskedTensor); ok {
			dmask := md.Mask()
			smask := ms.Mask()
			if cap(dmask) < len(smask) {
				dmask = make([]bool, len(smask))
				copy(dmask, md.Mask())
				md.SetMask(dmask)
			}
			copy(dmask, smask)
		}
	}

	e := src.Engine()
	if err := e.Memcpy(dst.arrPtr(), src.arrPtr()); err != nil {
		panic(err)
	}
	return dst.len()

	// return copyArray(dst.arr(), src.arr())
}

// copyDenseSliced copies a DenseTensor, but both are sliced
func copyDenseSliced(dst DenseTensor, dstart, dend int, src DenseTensor, sstart, send int) int {
	if dst.Dtype() != src.Dtype() {
		panic("Cannot copy DenseTensors of different types")
	}

	if ms, ok := src.(MaskedTensor); ok && ms.IsMasked() {
		if md, ok := dst.(MaskedTensor); ok {
			dmask := md.Mask()
			smask := ms.Mask()
			if cap(dmask) < dend {
				dmask = make([]bool, dend)
				copy(dmask, md.Mask())
				md.SetMask(dmask)
			}
			copy(dmask[dstart:dend], smask[sstart:send])
		}
	}
	if e := src.Engine(); e != nil {
		d := dst.arr().slice(dstart, dend)
		s := src.arr().slice(sstart, send)
		if err := e.Memcpy(&d, &s); err != nil {
			panic(err)
		}
		return d.Len()
	}
	return copyArraySliced(dst.arr(), dstart, dend, src.arr(), sstart, send)
}

// copyDenseIter copies a DenseTensor, with iterator
func copyDenseIter(dst, src DenseTensor, diter, siter Iterator) (int, error) {
	if dst.Dtype() != src.Dtype() {
		panic("Cannot copy Dense arrays of different types")
	}

	// if they all don't need iterators, and have the same data order
	if !dst.RequiresIterator() && !src.RequiresIterator() && dst.DataOrder().HasSameOrder(src.DataOrder()) {
		return copyDense(dst, src), nil
	}

	if !dst.IsNativelyAccessible() {
		return 0, errors.Errorf(inaccessibleData, dst)
	}
	if !src.IsNativelyAccessible() {
		return 0, errors.Errorf(inaccessibleData, src)
	}

	if diter == nil {
		diter = FlatIteratorFromDense(dst)
	}
	if siter == nil {
		siter = FlatIteratorFromDense(src)
	}

	// if it's a masked tensor, we copy the mask as well
	if ms, ok := src.(MaskedTensor); ok && ms.IsMasked() {
		if md, ok := dst.(MaskedTensor); ok {
			dmask := md.Mask()
			smask := ms.Mask()
			if cap(dmask) < len(smask) {
				dmask = make([]bool, len(smask))
				copy(dmask, md.Mask())
				md.SetMask(dmask)
			}
			copy(dmask, smask)
		}
	}
	return storage.CopyIter(dst.rtype(), dst.hdr(), src.hdr(), diter, siter), nil
}

func getPointer(a interface{}) unsafe.Pointer {
	switch at := a.(type) {
	case Memory:
		return at.Pointer()
	case bool:
		return unsafe.Pointer(&at)
	case int:
		return unsafe.Pointer(&at)
	case int8:
		return unsafe.Pointer(&at)
	case int16:
		return unsafe.Pointer(&at)
	case int32:
		return unsafe.Pointer(&at)
	case int64:
		return unsafe.Pointer(&at)
	case uint:
		return unsafe.Pointer(&at)
	case uint8:
		return unsafe.Pointer(&at)
	case uint16:
		return unsafe.Pointer(&at)
	case uint32:
		return unsafe.Pointer(&at)
	case uint64:
		return unsafe.Pointer(&at)
	case float32:
		return unsafe.Pointer(&at)
	case float64:
		return unsafe.Pointer(&at)
	case complex64:
		return unsafe.Pointer(&at)
	case complex128:
		return unsafe.Pointer(&at)
	case string:
		return unsafe.Pointer(&at)
	case uintptr:
		return unsafe.Pointer(at)
	case unsafe.Pointer:
		return at

		// POINTERS

	case *bool:
		return unsafe.Pointer(at)
	case *int:
		return unsafe.Pointer(at)
	case *int8:
		return unsafe.Pointer(at)
	case *int16:
		return unsafe.Pointer(at)
	case *int32:
		return unsafe.Pointer(at)
	case *int64:
		return unsafe.Pointer(at)
	case *uint:
		return unsafe.Pointer(at)
	case *uint8:
		return unsafe.Pointer(at)
	case *uint16:
		return unsafe.Pointer(at)
	case *uint32:
		return unsafe.Pointer(at)
	case *uint64:
		return unsafe.Pointer(at)
	case *float32:
		return unsafe.Pointer(at)
	case *float64:
		return unsafe.Pointer(at)
	case *complex64:
		return unsafe.Pointer(at)
	case *complex128:
		return unsafe.Pointer(at)
	case *string:
		return unsafe.Pointer(at)
	case *uintptr:
		return unsafe.Pointer(*at)
	case *unsafe.Pointer:
		return *at
	}

	panic("Cannot get pointer")
}

// scalarToHeader creates a Header from a scalar value
func scalarToHeader(a interface{}) *storage.Header {
	hdr := borrowHeader()
	hdr.Ptr = getPointer(a)
	hdr.L = 1
	hdr.C = 1
	return hdr
}
