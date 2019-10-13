package tensor

import (
	"fmt"

	"github.com/pkg/errors"
)

// An AP is an access pattern. It tells the various ndarrays how to access their data through the use of strides
// Through the AP, there are several definitions of things, most notably there are two very specific "special cases":
//		Scalar has Dims() of 0. However, its shape can take several forms:
//			- (1, 1)
//			- (1)
//		Vector has Dims() of 1, but its shape can take several forms:
//			- (x, 1)
//			- (1, x)
//			- (x)
//		Matrix has Dims() of 2. This is the most basic form. The len(shape) has to be equal to 2 as well
//		ndarray has Dims() of n.
type AP struct {
	shape   Shape // len(shape) is the operational definition of the dimensions
	strides []int // strides is usually calculated from shape
	fin     bool  // is this struct change-proof?

	o DataOrder
	Δ Triangle
}

func makeAP(size int) AP {
	return AP{
		shape:   Shape(BorrowInts(size)),
		strides: BorrowInts(size),
	}
}

// MakeAP creates an AP, given the shape and strides.
func MakeAP(shape Shape, strides []int, o DataOrder, Δ Triangle) AP {
	return AP{
		shape:   shape,
		strides: strides,
		o:       o,
		Δ:       Δ,
		fin:     true,
	}
}

// Init initializes an already created AP with a shape and stries.
// It will panic if AP is nil.
func (ap *AP) Init(shape Shape, strides []int) {
	ap.shape = shape
	ap.strides = strides
	ap.fin = true
}

// SetShape is for very specific times when modifying the AP is necessary, such as reshaping and doing I/O related stuff
//
// Caveats:
//
// - SetShape will recalculate the strides.
//
// - If the AP is locked, nothing will happen
func (ap *AP) SetShape(s ...int) {
	if !ap.fin {
		// scalars are a special case, we don't want to remove it completely
		if len(s) == 0 {
			if ap.shape == nil || ap.strides == nil {
				ap.shape = Shape{}
			}
			ap.shape = ap.shape[:0]
			ap.strides = ap.strides[:0]
			return
		}

		if ap.shape != nil {
			ReturnInts(ap.shape)
			ap.shape = nil
		}
		if ap.strides != nil {
			ReturnInts(ap.strides)
			ap.strides = nil
		}
		ap.shape = Shape(s).Clone()
		ap.strides = ap.calcStrides()
	}
}

// Shape returns the shape of the AP
func (ap *AP) Shape() Shape { return ap.shape }

// Strides returns the strides of the AP
func (ap *AP) Strides() []int { return ap.strides }

// Dims returns the dimensions of the shape in the AP
func (ap *AP) Dims() int { return ap.shape.Dims() }

// Size returns the expected array size of the shape
func (ap *AP) Size() int { return ap.shape.TotalSize() }

// String implements fmt.Stringer and runtime.Stringer
func (ap *AP) String() string { return fmt.Sprintf("%v", ap) }

// Format implements fmt.Formatter
func (ap *AP) Format(state fmt.State, c rune) {
	fmt.Fprintf(state, "Shape: %v, Stride: %v, Lock: %t", ap.shape, ap.strides, ap.fin)
}

// IsVector returns whether the access pattern falls into one of three possible definitions of vectors:
//		vanilla vector (not a row or a col)
//		column vector
//		row vector
func (ap *AP) IsVector() bool { return ap.shape.IsVector() }

// IsColVec returns true when the access pattern has the shape (x, 1)
func (ap *AP) IsColVec() bool { return ap.shape.IsColVec() }

// IsRowVec returns true when the access pattern has the shape (1, x)
func (ap *AP) IsRowVec() bool { return ap.shape.IsRowVec() }

// IsScalar returns true if the access pattern indicates it's a scalar value
func (ap *AP) IsScalar() bool { return ap.shape.IsScalar() }

// IsMatrix returns true if it's a matrix. This is mostly a convenience method. RowVec and ColVecs are also considered matrices
func (ap *AP) IsMatrix() bool { return len(ap.shape) == 2 }

// IsZero tell us if the ap has zero size
func (ap *AP) IsZero() bool {
	return len(ap.shape) == 0 && len(ap.strides) == 0 && !ap.fin && ap.o == 0 && ap.Δ == 0
}

// Zero zeros out an AP.
func (ap *AP) zero() {
	// log.Printf("ZEROING. Called by %v", string(debug.Stack()))

	// Jorge's original implementation for zeroing a AP is as below
	// but to cater for the (*Dense).fix() method of the *Dense
	// a nil shape is used to signal unsetness
	// so we cannot just truncate the shape even though it would be a lot more efficient

	// ap.shape = ap.shape[:0]
	// ap.strides = ap.strides[:0]
	ReturnInts([]int(ap.shape))
	ReturnInts(ap.strides)
	ap.zeroOnly()
}

// side effect free zeroing
func (ap *AP) zeroOnly() {
	ap.shape = nil
	ap.strides = nil

	ap.fin = false
	ap.o = 0
	ap.Δ = 0
}

func (ap *AP) zeroWithDims(dims int) {
	//ap.shape = BorrowInts(dims)
	//ap.strides = BorrowInts(dims)
	if cap(ap.shape) >= dims {
		ap.shape = ap.shape[:dims]
	}
	ap.shape = BorrowInts(dims)
	if cap(ap.strides) >= dims {
		ap.strides = ap.strides[:dims]
	}
	ap.strides = BorrowInts(dims)
}

// Clone clones the *AP. Clearly. It returns AP
func (ap *AP) Clone() (retVal AP) {
	retVal = makeAP(cap(ap.shape))

	copy(retVal.shape, ap.shape)
	copy(retVal.strides, ap.strides)

	// handle vectors
	retVal.shape = retVal.shape[:len(ap.shape)]
	retVal.strides = retVal.strides[:len(ap.strides)]

	retVal.fin = ap.fin
	retVal.o = ap.o
	retVal.Δ = ap.Δ
	return
}

func (ap *AP) CloneTo(dest *AP) {
	dest.shape = append(dest.shape[:0], ap.shape...)
	dest.strides = append(dest.strides[:0], ap.strides...)
	dest.fin = ap.fin
	dest.o = ap.o
	dest.Δ = ap.Δ
}

// DataOrder returns the data order of the AP.
func (ap *AP) DataOrder() DataOrder { return ap.o }

// C returns true if the access pattern is C-contiguous array
func (ap *AP) C() bool { return ap.o.IsRowMajor() && ap.o.IsContiguous() }

// F returns true if the access pattern is Fortran contiguous array
func (ap *AP) F() bool { return ap.o.IsColMajor() && ap.o.IsContiguous() }

// S returns the metadata of the sliced tensor.
func (ap *AP) S(size int, slices ...Slice) (newAP AP, ndStart, ndEnd int, err error) {
	if len(slices) > len(ap.shape) {
		// error
		err = errors.Errorf(dimMismatch, len(ap.shape), len(slices))
		return
	}

	ndEnd = size
	newShape := ap.shape.Clone()   // the new shape
	dims := ap.Dims()              // reported dimensions
	newStrides := BorrowInts(dims) // the new strides

	var outerDim int
	order := ap.o
	if ap.o.IsRowMajor() || ap.IsVector() {
		outerDim = 0
	} else {
		outerDim = len(ap.shape) - 1
	}

	for i := 0; i < dims; i++ {
		var sl Slice
		if i <= len(slices)-1 {
			sl = slices[i]
		}

		size := ap.shape[i]
		var stride int
		stride = ap.strides[i]
		// if ap.IsVector() {
		// 	// handles non-vanilla vectors
		// 	stride = ap.strides[0]
		// } else {
		// 	stride = ap.strides[i]
		// }

		var start, end, step int
		if start, end, step, err = SliceDetails(sl, size); err != nil {
			err = errors.Wrapf(err, "Unable to get slice details on slice %d with size %d: %v", i, sl, size)
			return
		}

		// a slice where start == end is []
		ndStart = ndStart + start*stride
		ndEnd = ndEnd - (size-end)*stride
		if step > 0 {
			newShape[i] = (end - start) / step
			newStrides[i] = stride * step

			//fix
			if newShape[i] <= 0 {
				newShape[i] = 1
			}
		} else {
			newShape[i] = (end - start)
			newStrides[i] = stride
		}

		if (sl != nil && (!ap.IsVector() && i != outerDim)) || step > 1 {
			order = MakeDataOrder(order, NonContiguous)
		}
	}

	if ndEnd-ndStart == 1 {
		// scalars are a special case
		newAP = AP{}
		newAP.SetShape() // make it a Scalar
		newAP.lock()
	} else {

		// drop any dimension with size 1, except the last dimension
		offset := 0
		for d := 0; d < dims; d++ {
			if newShape[d] == 1 && offset+d <= len(slices)-1 && slices[offset+d] != nil /*&& d != t.dims-1  && dims > 2*/ {
				newShape = append(newShape[:d], newShape[d+1:]...)
				newStrides = append(newStrides[:d], newStrides[d+1:]...)
				d--
				dims--
				offset++
			}
		}
		newAP = MakeAP(newShape, newStrides, order, ap.Δ)
	}
	return
}

// T returns the transposed metadata based on the given input
func (ap *AP) T(axes ...int) (retVal AP, a []int, err error) {
	// prep axes
	if len(axes) > 0 && len(axes) != ap.Dims() {
		err = errors.Errorf(dimMismatch, ap.Dims(), len(axes))
		return
	}

	dims := len(ap.shape)
	if len(axes) == 0 || axes == nil {
		axes = make([]int, dims)
		for i := 0; i < dims; i++ {
			axes[i] = dims - 1 - i
		}
	}
	a = axes

	// if axes is 0, 1, 2, 3... then no op
	if monotonic, incr1 := IsMonotonicInts(axes); monotonic && incr1 && axes[0] == 0 {
		return ap.Clone(), a, noopError{}
	}

	currentShape := ap.shape
	currentStride := ap.strides
	shape := make(Shape, len(currentShape))
	strides := make([]int, len(currentStride))

	switch {
	case ap.IsScalar():
		return
	case ap.IsVector():
		if axes[0] == 0 {
			return
		}
		copy(strides, currentStride)
		shape[0], shape[1] = currentShape[1], currentShape[0]
	default:
		copy(shape, currentShape)
		copy(strides, currentStride)
		err = UnsafePermute(axes, shape, strides)
		if err != nil {
			err = handleNoOp(err)
		}
	}

	o := MakeDataOrder(ap.o, Transposed)
	retVal = MakeAP(shape, strides, o, ap.Δ)
	retVal.fin = true
	return
}

// locking and unlocking is used to ensure that the shape and stride doesn't change (it's not really safe though, as a direct mutation of the strides/shape would still mutate it, but at least the dimensions cannot change)
func (ap *AP) lock()   { ap.fin = true }
func (ap *AP) unlock() { ap.fin = false }

func (ap *AP) calcStrides() []int {
	switch {
	case ap.o.IsRowMajor():
		return ap.shape.CalcStrides()
	case ap.o.IsColMajor():
		return ap.shape.CalcStridesColMajor()
	}
	panic("unreachable")
}

// setDataOrder is a method such that any tensor that embeds *AP will have the same method
func (ap *AP) setDataOrder(o DataOrder) {
	if !o.HasSameOrder(ap.o) {
		ap.o = ap.o.toggleColMajor()
	}
}

// TransposeIndex returns the new index given the old index
func TransposeIndex(i int, oldShape, pattern, oldStrides, newStrides []int) int {
	oldCoord, err := Itol(i, oldShape, oldStrides)
	if err != nil {
		panic(err) // or return error?
	}
	/*
		coordss, _ := Permute(pattern, oldCoord)
		coords := coordss[0]
		index, _ := Ltoi(newShape, strides, coords...)
	*/

	// The above is the "conceptual" algorithm.
	// Too many checks above slows things down, so the below is the "optimized" edition
	var index int
	for i, axis := range pattern {
		index += oldCoord[axis] * newStrides[i]
	}
	return index
}

// UntransposeIndex returns the old index given the new index
func UntransposeIndex(i int, oldShape, pattern, oldStrides, newStrides []int) int {
	newPattern := make([]int, len(pattern))
	for i, p := range pattern {
		newPattern[p] = i
	}
	return TransposeIndex(i, oldShape, newPattern, oldStrides, newStrides)
}

// BroadcastStrides handles broadcasting from different shapes.
//
// Deprecated: this function will be unexported
func BroadcastStrides(destShape, srcShape Shape, destStrides, srcStrides []int) (retVal []int, err error) {
	dims := len(destShape)
	start := dims - len(srcShape)

	if destShape.IsVector() && srcShape.IsVector() {
		return []int{srcStrides[0]}, nil
	}

	if start < 0 {
		//error
		err = errors.Errorf(dimMismatch, dims, len(srcShape))
		return
	}

	retVal = BorrowInts(len(destStrides))
	for i := dims - 1; i >= start; i-- {
		s := srcShape[i-start]
		switch {
		case s == 1:
			retVal[i] = 0
		case s != destShape[i]:
			// error
			err = errors.Errorf("Cannot broadcast from %v to %v", srcShape, destShape)
			return
		default:
			retVal[i] = srcStrides[i-start]
		}
	}
	for i := 0; i < start; i++ {
		retVal[i] = 0
	}
	return
}
