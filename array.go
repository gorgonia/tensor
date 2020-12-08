package tensor

import (
	"fmt"
	"reflect"
	"sync"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"
)

// array is the underlying generic array.
type array struct {
	storage.Header       // the header - the Go representation (a slice)
	t              Dtype // the element type
}

// makeArray makes an array. The memory allocation is handled by Go
func makeArray(t Dtype, length int) array {
	v := malloc(t, length)
	hdr := storage.Header{
		Raw: v,
	}
	return array{
		Header: hdr,
		t:      t,
	}

}

// arrayFromSlice creates an array from a slice. If x is not a slice, it will panic.
func arrayFromSlice(x interface{}) array {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	elT := xT.Elem()

	return array{
		Header: storage.Header{
			Raw: storage.AsByteSlice(x),
		},
		t: Dtype{elT},
	}
}

func (a *array) Len() int { return a.Header.TypedLen(a.t.Type) }

func (a *array) Cap() int { return a.Header.TypedLen(a.t.Type) }

// fromSlice populates the value from a slice
func (a *array) fromSlice(x interface{}) {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	elT := xT.Elem()
	a.Raw = storage.AsByteSlice(x)
	a.t = Dtype{elT}
}

// fromSliceOrTensor populates the value from a slice or anything that can form an array
func (a *array) fromSliceOrArrayer(x interface{}) {
	if T, ok := x.(arrayer); ok {
		xp := T.arrPtr()

		// if the underlying array hasn't been allocated, or not enough has been allocated
		if a.Header.Raw == nil {
			a.Header.Raw = malloc(xp.t, xp.Len())
		}

		a.t = xp.t
		copyArray(a, T.arrPtr())
		return
	}
	a.fromSlice(x)
}

// byteSlice casts the underlying slice into a byte slice. Useful for copying and zeroing, but not much else
func (a array) byteSlice() []byte { return a.Header.Raw }

// sliceInto creates a slice. Instead of returning an array, which would cause a lot of reallocations, sliceInto expects a array to
// already have been created. This allows repetitive actions to be done without having to have many pointless allocation
func (a *array) sliceInto(i, j int, res *array) {
	c := a.Cap()

	if i < 0 || j < i || j > c {
		panic(fmt.Sprintf("Cannot slice %v - index %d:%d is out of bounds", a, i, j))
	}

	s := i * int(a.t.Size())
	e := j * int(a.t.Size())
	c = c - i

	res.Raw = a.Raw[s:e]

}

// slice slices an array
func (a array) slice(start, end int) array {
	if end > a.Len() {
		panic("Index out of range")
	}
	if end < start {
		panic("Index out of range")
	}

	s := start * int(a.t.Size())
	e := end * int(a.t.Size())

	return array{
		Header: storage.Header{Raw: a.Raw[s:e]},
		t:      a.t,
	}
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
func (a *array) Uintptr() uintptr { return uintptr(unsafe.Pointer(&a.Header.Raw[0])) }

// MemSize returns how big the slice is in bytes
func (a *array) MemSize() uintptr { return uintptr(len(a.Header.Raw)) }

// Data returns the representation of a slice.
func (a array) Data() interface{} {
	// build a type of []T
	shdr := reflect.SliceHeader{
		Data: a.Uintptr(),
		Len:  a.Len(),
		Cap:  a.Cap(),
	}
	sliceT := reflect.SliceOf(a.t.Type)
	ptr := unsafe.Pointer(&shdr)
	val := reflect.Indirect(reflect.NewAt(sliceT, ptr))
	return val.Interface()

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

	l := a.Len()
	for i := 0; i < l; i++ {
		val := reflect.NewAt(a.t.Type, storage.ElementAt(i, unsafe.Pointer(&a.Header.Raw[0]), a.t.Size()))
		val = reflect.Indirect(val)
		val.Set(reflect.Zero(a.t))
	}
}

func (a *array) hdr() *storage.Header { return &a.Header }
func (a *array) rtype() reflect.Type  { return a.t.Type }

/* MEMORY MOVEMENT STUFF */

// malloc is standard Go allocation of a block of memory - the plus side is that Go manages the memory
func malloc(t Dtype, length int) []byte {
	size := int(calcMemSize(t, length))
	return make([]byte, size)
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
		darr := dst.arr()
		sarr := src.arr()
		da := darr.slice(dstart, dend)
		sa := sarr.slice(sstart, send)

		switch e.(type) {
		case NonStdEngine:
			if err := e.Memcpy(&da, &sa); err != nil {
				panic(err)
			}
		default:
			// THIS IS AN OPTIMIZATION. REVISIT WHEN NEEDED.
			//
			// THE PURPOSE of this optimization is to make this perform better under
			// default circumstances.
			//
			// The original code simply uses t.Engine().Memcpy(&dSlice, &tSlice).
			// A variant can still be seen in the NonStdEngine case above.
			//
			// The `array.slice()` method has been optimized to return `array2`, which is a
			// non-heap allocated type.
			// a value of `array2` cannot have its address taken - e.g.
			// 	var a array2
			// 	doSomething(&a) // ← this cannot be done
			//
			// We *could* make `array2` implement Memory. But then a lot of runtime.convT2I and
			// runtime.convI2T would be called. Which defeats the purpose of making things fast.
			//
			// So instead, we check to see if the Engine uses standard allocation methods.
			// Typically this means `StdEng`.
			//
			// If so, we directly use storage.Copy instead of using the engine
			storage.Copy(da.t.Type, &da.Header, &sa.Header)
		}

		return da.Len()
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

type scalarPtrCount struct {
	Ptr   unsafe.Pointer
	Count int
}

// scalarRCLock is a lock for the reference counting list.
var scalarRCLock sync.Mutex

// scalarRC is a bunch of reference counted pointers to scalar values
var scalarRC = make(map[uintptr]*sync.Pool) // uintptr is the size, the pool stores []byte

func scalarPool(size uintptr) *sync.Pool {
	scalarRCLock.Lock()
	pool, ok := scalarRC[size]
	if !ok {
		pool = &sync.Pool{
			New: func() interface{} { return make([]byte, size) },
		}
		scalarRC[size] = pool
	}
	scalarRCLock.Unlock()
	return pool
}

func allocScalar(a interface{}) []byte {
	atype := reflect.TypeOf(a)
	size := atype.Size()
	pool := scalarPool(size)
	return pool.Get().([]byte)
}

func freeScalar(bs []byte) {
	if bs == nil {
		return
	}

	// zero out
	for i := range bs {
		bs[i] = 0
	}

	size := uintptr(len(bs))

	// put it back into pool
	pool := scalarPool(size)
	pool.Put(bs)
}

// scalarToHeader creates a Header from a scalar value
func scalarToHeader(a interface{}) (hdr *storage.Header, newAlloc bool) {
	var raw []byte
	switch at := a.(type) {
	case Memory:
		raw = storage.FromMemory(at.Uintptr(), at.MemSize())
	default:
		raw = allocScalar(a)
		newAlloc = true
	}
	hdr = borrowHeader()
	hdr.Raw = raw
	if newAlloc {
		copyScalarToPrealloc(a, hdr.Raw)
	}

	return hdr, newAlloc
}

func copyScalarToPrealloc(a interface{}, bs []byte) {
	xV := reflect.ValueOf(a)
	xT := reflect.TypeOf(a)

	p := unsafe.Pointer(&bs[0])
	v := reflect.NewAt(xT, p)
	reflect.Indirect(v).Set(xV)
	return
}
