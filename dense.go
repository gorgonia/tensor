package tensor

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/tensor/internal/storage"
)

const (
	maskCompEvery int = 8
)

// Dense represents a dense tensor - this is the most common form of tensors. It can be used to represent vectors, matrices.. etc
type Dense struct {
	AP
	array

	flag MemoryFlag
	e    Engine         // execution engine for the *Dense
	oe   StandardEngine // optimized engine

	// backup AP. When a transpose is done, the old *AP is backed up here, for easy untransposes
	old           AP
	transposeWith []int

	// if viewOf != nil, then this *Dense is a view.
	viewOf uintptr

	mask       []bool // mask slice can be used to identify missing or invalid values. len(mask)<=len(v)
	maskIsSoft bool
}

// NewDense creates a new *Dense. It tries its best to get from the tensor pool.
func NewDense(dt Dtype, shape Shape, opts ...ConsOpt) *Dense {
	return recycledDense(dt, shape, opts...)
}

func recycledDense(dt Dtype, shape Shape, opts ...ConsOpt) (retVal *Dense) {
	retVal = recycledDenseNoFix(dt, shape, opts...)
	retVal.fix()
	if err := retVal.sanity(); err != nil {
		panic(err)
	}
	return
}

func recycledDenseNoFix(dt Dtype, shape Shape, opts ...ConsOpt) (retVal *Dense) {
	//	size := shape.TotalSize()
	//if shape.IsScalar() {
	//	size = 1
	//}
	retVal = borrowDense()
	retVal.array.t = dt
	retVal.AP.zeroWithDims(shape.Dims())

	for _, opt := range opts {
		opt(retVal)
	}
	retVal.setShape(shape...)
	return
}

func (t *Dense) fromSlice(x interface{}) {
	t.array.Header.Raw = nil // GC anything else
	t.array.fromSlice(x)
}

func (t *Dense) addMask(mask []bool) {
	l := len(mask)
	if l > 0 && l != t.len() {
		panic("Mask is not same length as data")
	}
	t.mask = mask
}

func (t *Dense) makeArray(size int) {
	switch te := t.e.(type) {
	case NonStdEngine:
		t.flag = MakeMemoryFlag(t.flag, ManuallyManaged)
	case arrayMaker:
		te.makeArray(&t.array, t.t, size)
		return
	case StandardEngine:
	default:

	}

	memsize := calcMemSize(t.t, size)
	mem, err := t.e.Alloc(memsize)
	if err != nil {
		panic(err)
	}

	t.array.Raw = storage.FromMemory(mem.Uintptr(), uintptr(memsize))
	return
}

// Info returns the access pattern which explains how the data in the underlying array is accessed. This is mostly used for debugging.
func (t *Dense) Info() *AP { return &t.AP }

// Dtype returns the data type of the *Dense tensor.
func (t *Dense) Dtype() Dtype { return t.t }

// Data returns the underlying array. If the *Dense represents a scalar value, the scalar value is returned instead
func (t *Dense) Data() interface{} {
	if t.IsScalar() {
		return t.Get(0)
	}

	// build a type of []T
	shdr := reflect.SliceHeader{
		Data: t.array.Uintptr(),
		Len:  t.array.Len(),
		Cap:  t.array.Cap(),
	}
	sliceT := reflect.SliceOf(t.t.Type)
	ptr := unsafe.Pointer(&shdr)
	val := reflect.Indirect(reflect.NewAt(sliceT, ptr))
	return val.Interface()
}

// DataSize returns the size of the underlying array. Typically t.DataSize() == t.Shape().TotalSize()
func (t *Dense) DataSize() int {
	if t.IsScalar() {
		return 0 // DOUBLE CHECK
	}
	return t.array.Len()
}

// Engine returns the execution engine associated with this Tensor
func (t *Dense) Engine() Engine { return t.e }

// Reshape reshapes a *Dense. If the tensors need to be materialized (either it's a view or transpose), it will be materialized before the reshape happens
func (t *Dense) Reshape(dims ...int) error {
	if t.Shape().TotalSize() != Shape(dims).TotalSize() {
		return errors.Errorf("Cannot reshape %v into %v", t.Shape(), dims)
	}

	if t.viewOf != 0 && t.o.IsNotContiguous() {
		return errors.Errorf(methodNYI, "Reshape", "non-contiguous views")
	}

	if !t.old.IsZero() {
		t.Transpose()
	}

	return t.reshape(dims...)
}

func (t *Dense) reshape(dims ...int) error {
	t.setShape(dims...)
	return t.sanity()
}

func (t *Dense) unsqueeze(axis int) error {
	if axis > t.shape.Dims()+1 {
		return errors.Errorf("Cannot unsqueeze on axis %d when the tensor has shape %v", axis, t.shape)
	}
	t.shape = append(t.shape, 1)
	copy(t.shape[axis+1:], t.shape[axis:])
	t.shape[axis] = 1

	t.strides = append(t.strides, 1)
	copy(t.strides[axis+1:], t.strides[axis:])

	return nil
}

// ScalarValue returns the scalar value of a *Tensor,
// IF and ONLY IF it's a Tensor representation of a scalar value.
// This is required because operations like a (vec · vec) would return a scalar value.
// I didn't want to return interface{} for all the API methods, so the next best solution is to
// wrap the scalar value in a *Tensor
func (t *Dense) ScalarValue() interface{} {
	if !t.IsScalar() {
		panic(fmt.Sprintf("ScalarValue only works when the Tensor is a representation of a scalar value. The value of the tensor is %v", t))
	}

	return t.Get(0)
}

// IsManuallyManaged returns true if the memory associated with this *Dense is manually managed (by the user)
func (t *Dense) IsManuallyManaged() bool { return t.flag.manuallyManaged() }

// IsNativelyAccessible checks if the pointers are accessible by Go
func (t *Dense) IsNativelyAccessible() bool { return t.flag.nativelyAccessible() }

// Clone clones a *Dense. It creates a copy of the data, and the underlying array will be allocated
func (t *Dense) Clone() interface{} {
	if t.e != nil {
		retVal := new(Dense)
		t.AP.CloneTo(&retVal.AP)
		retVal.t = t.t
		retVal.e = t.e
		retVal.oe = t.oe
		retVal.flag = t.flag
		retVal.makeArray(t.Len())

		if !t.old.IsZero() {
			retVal.old = t.old.Clone()
			t.old.CloneTo(&retVal.old)
		}
		copyDense(retVal, t)
		retVal.lock()

		return retVal
	}
	panic("Unreachable: No engine")
}

// IsMasked indicates whether tensor is masked
func (t *Dense) IsMasked() bool { return len(t.mask) == t.len() }

// MaskFromDense adds a mask slice to tensor by XORing dense arguments' masks
func (t *Dense) MaskFromDense(tts ...*Dense) {
	hasMask := BorrowBools(len(tts))
	defer ReturnBools(hasMask)

	numMasked := 0
	var masked = false

	for i, tt := range tts {
		if tt != nil {
			hasMask[i] = tt.IsMasked()
			masked = masked || hasMask[i]
			if hasMask[i] {
				numMasked++
			}
		}
	}
	if numMasked < 1 {
		return
	}

	//Only make mask if none already. This way one of the tts can be t itself

	if len(t.mask) < t.DataSize() {
		t.makeMask()
	}

	for i, tt := range tts {
		if tt != nil {
			n := len(tt.mask)
			if hasMask[i] {
				for j := range t.mask {
					t.mask[j] = t.mask[j] || tt.mask[j%n]
				}
			}
		}
	}
}

// Private methods

func (t *Dense) cap() int       { return t.array.Cap() }
func (t *Dense) len() int       { return t.array.Len() } // exactly the same as DataSize
func (t *Dense) arr() array     { return t.array }
func (t *Dense) arrPtr() *array { return &t.array }

func (t *Dense) setShape(s ...int) {
	t.unlock()
	t.SetShape(s...)
	t.lock()
	return
}

func (t *Dense) setAP(ap *AP) { t.AP = *ap }

func (t *Dense) fix() {
	if t.e == nil {
		t.e = StdEng{}
	}

	if oe, ok := t.e.(StandardEngine); ok {
		t.oe = oe
	}

	switch {
	case t.IsScalar() && t.array.Header.Raw == nil:
		t.makeArray(1)
	case t.Shape() == nil && t.array.Header.Raw != nil:
		size := t.Len()
		if size == 1 {
			t.SetShape() // scalar
		} else {
			t.SetShape(size) // vector
		}
	case t.array.Header.Raw == nil && t.t != Dtype{}:
		size := t.Shape().TotalSize()
		t.makeArray(size)

	}
	if len(t.mask) != t.len() {
		t.mask = t.mask[:0]
	}
	t.lock() // don't put this in a defer - if t.array.Ptr == nil and t.Shape() == nil. then leave it unlocked
}

// makeMask adds a mask slice to tensor if required
func (t *Dense) makeMask() {
	var size int
	size = t.shape.TotalSize()
	if len(t.mask) >= size {
		t.mask = t.mask[:size]
	}
	if cap(t.mask) < size {
		t.mask = make([]bool, size)
	}
	t.mask = t.mask[:size]
	memsetBools(t.mask, false)
}

// sanity is a function that sanity checks that a tensor is correct.
func (t *Dense) sanity() error {
	if !t.AP.IsZero() && t.Shape() == nil && t.array.Header.Raw == nil {
		return errors.New(emptyTensor)
	}

	size := t.Len()
	expected := t.Size()
	if t.viewOf == 0 && size != expected && !t.IsScalar() {
		return errors.Wrap(errors.Errorf(shapeMismatch, t.Shape(), size), "sanity check failed")
	}

	// TODO: sanity check for views
	return nil
}

// isTransposed returns true if the *Dense holds a transposed array.
func (t *Dense) isTransposed() bool { return t.old.IsZero() }

// oshape returns the original shape
func (t *Dense) oshape() Shape {
	if !t.old.IsZero() {
		return t.old.Shape()
	}
	return t.Shape()
}

// ostrides returns the original strides
func (t *Dense) ostrides() []int {
	if !t.old.IsZero() {
		return t.old.Strides()
	}
	return t.Strides()
}

// ShallowClone clones the *Dense without making a copy of the underlying array
func (t *Dense) ShallowClone() *Dense {
	retVal := borrowDense()
	retVal.e = t.e
	retVal.oe = t.oe
	t.AP.CloneTo(&retVal.AP)
	retVal.flag = t.flag
	retVal.array = t.array

	retVal.old = t.old
	retVal.transposeWith = t.transposeWith
	retVal.viewOf = t.viewOf
	retVal.mask = t.mask
	retVal.maskIsSoft = t.maskIsSoft
	return retVal
}

func (t *Dense) oldAP() *AP           { return &t.old }
func (t *Dense) setOldAP(ap *AP)      { t.old = *ap }
func (t *Dense) transposeAxes() []int { return t.transposeWith }

//go:nocheckptr
func (t *Dense) parentTensor() *Dense {
	if t.viewOf != 0 {
		return (*Dense)(unsafe.Pointer(t.viewOf))
	}
	return nil
}

func (t *Dense) setParentTensor(d *Dense) {
	if d == nil {
		t.viewOf = 0
		return
	}
	t.viewOf = uintptr(unsafe.Pointer(d))
}

/* ------ Mask operations */

//ResetMask fills the mask with either false, or the provided boolean value
func (t *Dense) ResetMask(val ...bool) error {
	if !t.IsMasked() {
		t.makeMask()
	}
	var fillValue = false
	if len(val) > 0 {
		fillValue = val[0]
	}
	memsetBools(t.mask, fillValue)
	return nil
}

// HardenMask forces the mask to hard. If mask is hard, then true mask values can not be unset
func (t *Dense) HardenMask() bool {
	t.maskIsSoft = false
	return t.maskIsSoft
}

// SoftenMask forces the mask to soft
func (t *Dense) SoftenMask() bool {
	t.maskIsSoft = true
	return t.maskIsSoft
}

// MaskFromSlice makes mask from supplied slice
func (t *Dense) MaskFromSlice(x interface{}) {
	t.makeMask()
	n := len(t.mask)
	switch m := x.(type) {
	case []bool:
		copy(t.mask, m)
		return
	case []int:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []int8:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []int16:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []int32:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []int64:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []uint:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []byte:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []uint16:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []uint32:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []uint64:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []float32:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []float64:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []complex64:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []complex128:
		for i, v := range m {
			if v != 0 {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	case []string:
		for i, v := range m {
			if v != "" {
				t.mask[i] = true
			}
			if i >= n {
				return
			}
		}
	default:
		return
	}
}

// Memset sets all the values in the *Dense tensor.
func (t *Dense) Memset(x interface{}) error {
	if !t.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, t)
	}
	if t.RequiresIterator() {
		it := newFlatIterator(&t.AP)
		return t.array.memsetIter(x, it)
	}
	return t.array.Memset(x)
}

// Eq checks that any two things are equal. If the shapes are the same, but the strides are not the same, it's will still be considered the same
func (t *Dense) Eq(other interface{}) bool {
	if ot, ok := other.(*Dense); ok {
		if ot == t {
			return true
		}
		if !t.Shape().Eq(ot.Shape()) {
			return false
		}

		return t.array.Eq(&ot.array)
	}
	return false
}

func (t *Dense) Zero() {
	if t.RequiresIterator() {
		it := newFlatIterator(&t.AP)
		if err := t.zeroIter(it); err != nil {
			panic(err)
		}
	}
	if t.IsMasked() {
		t.ResetMask()
	}
	t.array.Zero()
}

func (t *Dense) Mask() []bool { return t.mask }

func (t *Dense) SetMask(mask []bool) {
	// if len(mask) != t.len() {
	// 	panic("Cannot set mask")
	// }
	t.mask = mask
}

func (t *Dense) slice(start, end int) {
	t.array = t.array.slice(start, end)
}

// RequiresIterator indicates if an iterator is required to read the data in *Dense in the correct fashion
func (t *Dense) RequiresIterator() bool {
	if t.len() == 1 {
		return false
	}
	// non continuous slice, transpose, or masked. If it's a slice and contiguous, then iterator is not required
	if !t.o.IsContiguous() || !t.old.IsZero() || t.IsMasked() {
		return true
	}
	return false
}

func (t *Dense) Iterator() Iterator { return IteratorFromDense(t) }

func (t *Dense) standardEngine() StandardEngine { return t.oe }
