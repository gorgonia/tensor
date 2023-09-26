package dense

import (
	"fmt"
	"unsafe"

	"gorgonia.org/dtype"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	"gorgonia.org/tensor/internal/flatiter"
	gutils "gorgonia.org/tensor/internal/utils"
)

var (
	_ tensor.Basic[float64]                   = &Dense[float64]{}
	_ tensor.Cloner[*Dense[float64]]          = &Dense[float64]{}
	_ tensor.Tensor[float64, *Dense[float64]] = &Dense[float64]{}
)

type Dense[DT any] struct {
	AP
	data  []DT
	bytes []byte // the original data, but as a slice of bytes

	// transposedWith is the indices of permutation for a transposition
	transposedWith []int

	f MemoryFlag
	e Engine
	t dtype.Dtype[DT]
}

// consFromDatatype is a special construction function to create a *Dense[???] where ??? is only known at runtime
func consFromDatatype(dt dtype.Datatype, opts ...ConsOpt) (retVal DescWithStorage, err error) {
	fn, ok := consRegistry[dt]
	if !ok {
		return nil, errors.Errorf("Dtype %T not supported", dt)
	}
	return fn(opts...)
}

func New[DT any](opts ...ConsOpt) *Dense[DT] {
	c := parseConsOpt[DT](opts...)
	if !c.IsOK() {
		panic("Construction options do not make sense Please check that either a backing array or a shape has been passed in; or if `AsFortran` was used, that a backing array was passed in as well")
	}
	return fromConstructor[DT](c)
}

func parseConsOpt[DT any](opts ...ConsOpt) *Constructor {
	c := new(Constructor)
	for _, o := range opts {
		o(c)
	}
	if c.Engine == nil {
		c.Engine = defaultEngine[DT]()
	}
	return c
}

func fromConstructor[DT any](c *Constructor) *Dense[DT] {
	var data []DT
	var shape shapes.Shape
	var dataorder DataOrder
	var memoryflag MemoryFlag
	if c.AsFortran {
		dataorder = internal.ColMajor
	}

	switch {
	case c.Shape != nil && c.Data == nil:
		// Shape provided, but data is not. Create data based on engine
		shape = c.Shape
		elements := c.Shape.TotalSize()
		data = makeBacking[DT](c.Engine, elements, dataorder)
	case c.Shape != nil && c.Data != nil:
		// Shape provided, and data provided. Check if data is memory
		shape = c.Shape
		data = handleData[DT](c.Engine, c.Data, dataorder)

		// handle flags.
		sz := shape.TotalSize()
		switch {
		case len(data) < sz:
			newData := constructionbehaviour(data, shape)
			data = handleData[DT](c.Engine, newData, dataorder)
		case len(data) > sz:
			// then we set the overallocated flag
			memoryflag |= internal.IsOverallocated
		default:
			// nothing doin'.

		}
	case c.Shape == nil && c.Data != nil:
		// Shape not provided, but data is. We'll create a new shape that is a flat shape
		data = handleData[DT](c.Engine, c.Data, dataorder)
		shape = shapes.Shape{len(data)}
	default:
		// no shape? panic
		panic("YYY")
	}

	return construct(data, shape, c.Engine, memoryflag)
}

func handleData[DT any](e Engine, d any, do DataOrder) (data []DT) {
	switch d := d.(type) {
	case []DT:
		// if engine is manually managed or natively inaccessible, then we need to allocate and copy
		// otherwise, can just use it as data
		data = d

		if e.WorksWith(internal.NativelyInaccessible|internal.ManuallyManaged, do) {
			// allocate and copy `d` into `alloc`
			// then get a slice from `alloc`
			mem := internal.SliceAsMemory(d)
			alloc, err := e.Alloc(int64(mem.MemSize()))
			if err != nil {
				panic(err)
			}
			if err := e.Memcpy(alloc, mem); err != nil {
				panic(err)
			}
			data = internal.SliceFromMemory[DT](alloc)
		}
	case Memory:
		// turn memory into a slice of DT
		data = internal.SliceFromMemory[DT](d)
	case DT:
		// from scalar - TODO: what if engine is managed?
		data = []DT{d}
		return
	default:
		var v DT
		panic(fmt.Sprintf("Cannot construct *Dense[%T]. The backing data is %T", v, d))
	}
	return
}

func makeBacking[DT any](e Engine, n int, dataorder DataOrder) []DT {
	if e.WorksWith(internal.NativelyInaccessible|internal.ManuallyManaged, dataorder) {
		// then allocate memory
		sz := internal.CalcMemSize[DT](n)
		mem, err := e.Alloc(sz)
		if err != nil {
			panic(err)
		}
		return internal.SliceFromMemory[DT](mem)

	}
	return make([]DT, n)
}

func construct[DT any](data []DT, shape shapes.Shape, e Engine, f MemoryFlag) *Dense[DT] {
	var ap AP
	ap.SetShape(shape...)
	dt := gutils.GetDtype[DT]()
	f |= memoryFlagFromEngine(e)
	return &Dense[DT]{
		AP:    ap,
		data:  data,
		bytes: gutils.BytesFromSlice(data),
		e:     e,
		f:     f,
		t:     dt,
	}

}

func (t *Dense[DT]) copyMetadata(srcAP AP, srcEng Engine, srcFlag MemoryFlag, srcDT dtype.Dtype[DT]) {
	t.AP = srcAP
	t.e = srcEng
	t.f = srcFlag
	t.t = srcDT
}

// restore restores any overallocated tensors to the correct length
func (t *Dense[T]) Restore() {
	if cap(t.data) > len(t.data) {
		t.data = t.data[:cap(t.data)]
	}
}

/* Construction-related methods */

// fix fixes flags and data and stuff.
func (t *Dense[T]) fix() {
	totalsize := t.Shape().TotalSize()
	switch {
	case totalsize < len(t.data):
		// overallocated
		t.f |= internal.IsOverallocated
		t.data = t.data[:totalsize]
		//t.bytes = t.bytes[:totalsize*int(t.t.Size())] // unsure if we ever need this
	case totalsize == cap(t.data):
		// exactly as allocated
		t.f &^= internal.IsOverallocated // clear Overallocated flag
		t.Restore()
	}

}

/* Getters for properties */

func (t *Dense[T]) Info() *AP             { return &t.AP }
func (t *Dense[T]) Dtype() dtype.Datatype { return t.t }
func (t *Dense[T]) Engine() Engine        { return t.e }
func (t *Dense[T]) Flags() MemoryFlag     { return t.f }

/* Data Access */

func (t *Dense[T]) Data() []T {
	if t == nil {
		return nil
	}
	f := t.Flags()
	switch {
	case t.IsScalarEquiv():
		return t.data[:1]
	case f.IsOverallocated() && !f.IsView():
		sz := t.Size()
		return t.data[:sz]
	case f.IsOverallocated() && f.IsView():
		panic("Not Yet Implemented: Overallocated tensors as view")
	default:
		return t.data
	}
}

func (t *Dense[T]) DataSize() int { return len(t.data) }

// ScalarValue returns the scalar equivalent of the tensor.
func (t *Dense[T]) ScalarValue() T { return t.data[0] }

func (t *Dense[T]) At(coords ...int) (T, error) {
	var empty T
	if !t.IsNativelyAccessible() {
		return empty, errors.Errorf(errors.InaccessibleData, t)
	}
	if len(coords) != t.Dims() {
		return empty, errors.Errorf(errors.DimMismatch, t.Dims(), len(coords))
	}
	at, err := tensor.Ltoi(t.Shape(), t.Strides(), coords...)
	if err != nil {
		return empty, errors.Wrap(err, "At()")
	}

	return t.data[at], nil

}

func (t *Dense[T]) Memset(v T) error {
	if t.f.IsNativelyAccessible() {
		for i := range t.data {
			t.data[i] = v
		}
		return nil
	}
	return t.e.Memset(t, v)
}

func (t *Dense[T]) SetAt(v T, coords ...int) error {
	if !t.IsNativelyAccessible() {
		return errors.Errorf(errors.InaccessibleData, t)
	}
	if len(coords) != t.Dims() {
		return errors.Errorf(errors.DimMismatch, t.Dims(), len(coords))
	}
	at, err := tensor.Ltoi(t.Shape(), t.Strides(), coords...)
	if err != nil {
		return errors.Wrap(err, "At()")
	}
	t.data[at] = v
	return nil
}

func (t *Dense[T]) Zero() {
	if !t.IsNativelyAccessible() {
		t.e.Memclr(t)
		return
	}
	var z T
	for i := range t.data {
		t.data[i] = z
	}

}

/* Construction related methods */

// FromDense returns itself. This allows `*Dense[T]` to implement `FromDensor[Dt]“
func (t *Dense[T]) FromDense(d *Dense[T]) *Dense[T] { return d }

// GetDense returns itself. This allows `*Dense[T]` to implement `Densor[T]`
func (t *Dense[T]) GetDense() *Dense[T] { return t }

// Alike creates a new *Dense[T]. If no options is passed in, it will create a *Dense[T] that has the same shape and strides and data length
// as the input. Otherwise it will simply call `New`
func (t *Dense[T]) Alike(opts ...ConsOpt) *Dense[T] {
	if len(opts) != 0 {
		return New[T](opts...)
	}
	data := makeBacking[T](t.e, len(t.data), t.AP.DataOrder())
	retVal := &Dense[T]{
		data:  data,
		bytes: gutils.BytesFromSlice(data),
	}
	retVal.copyMetadata(t.AP.Clone(), t.e, t.f, t.t) // the flag here is just a copy - no setting of IsView
	return retVal
}

func (t *Dense[T]) AlikeAsType(dt dtype.Datatype, opts ...ConsOpt) DescWithStorage {
	switch dt {
	case dtype.Bool:
		c := parseConsOpt[bool](opts...)
		return fromConstructor[bool](c)
		// TODO: engine alike too
		//return retVal
	case dtype.Int:
		c := parseConsOpt[int](opts...)
		return fromConstructor[int](c)
	default:
		panic("NYI: put a pull request")
	}

	return nil
}

func (t *Dense[T]) AlikeAsBasic(opts ...ConsOpt) tensor.Basic[T] { return t.Alike(opts...) }

func (t *Dense[T]) AlikeAsDescWithStorage(opts ...ConsOpt) tensor.DescWithStorage {
	return t.Alike(opts...)
}

func (t *Dense[T]) Clone() *Dense[T] {
	retVal := t.Alike()
	copy(retVal.data, t.data)
	return retVal
}

func (t *Dense[T]) ShallowClone() *Dense[T] {
	retVal := &Dense[T]{
		data:  t.data,
		bytes: t.bytes,
	}
	retVal.copyMetadata(t.AP.Clone(), t.e, t.f.ViewFlag(), t.t)
	return retVal
}

func (t *Dense[T]) CloneAsBasic() tensor.Basic[T] { return t.Clone() }

func (t *Dense[T]) RequiresIterator() bool {
	if len(t.data) == 1 {
		return false
	}
	// TODO

	// non contiguous?
	if !t.DataOrder().IsContiguous() {
		return true
	}

	if t.f.IsView() {
		return true
	}
	return false
}

func (t *Dense[T]) Iterator() Iterator {
	if t == nil {
		return nil
	}
	return flatiter.New(&t.AP)
}

func (t *Dense[T]) IsMaterializable() bool { return t.transposedWith == nil && !t.f.IsView() }

func (t *Dense[T]) Materialize() (*Dense[T], error) {
	if !t.IsMaterializable() {
		return t, nil
	}
	panic("NYI")
}

func (t *Dense[T]) IsNativelyAccessible() bool {
	if t == nil {
		return false
	}
	return t.f.IsNativelyAccessible()
}
func (t *Dense[T]) IsManuallyManaged() bool { return t.f.IsManuallyManaged() }
func (t *Dense[T]) IsView() bool            { return t.f.IsView() }
func (t *Dense[T]) MemSize() uintptr        { return uintptr(len(t.bytes)) }
func (t *Dense[T]) Uintptr() uintptr        { return uintptr(unsafe.Pointer(&t.bytes[0])) }

func (t *Dense[T]) Eq(u *Dense[T]) bool {
	// TODO: natively accessible stuff
	// TODO: fortran flag stuff

	if !t.Shape().Eq(u.Shape()) {
		return false
	}
	if !sliceEq(t.Strides(), u.Strides()) {
		return false
	}

	e, ok := t.e.(sliceEqer[T])
	if !ok {
		panic("Cannot compute whether the data is equal. Engine does not implement SliceEq method")
	}
	return e.SliceEq(t.data, u.data)
}
