package internal

import "strings"

// MemoryFlag is a flag representing the use possibilities of Memory
type MemoryFlag byte

const (
	// NativelyInaccessible indicates that the data in the memory cannot be accessed by Go code.
	NativelyInaccessible MemoryFlag = 1 << iota
	// ManuallyManaged indicates that the memory is managed by something else. Any Tensor with
	// manually managed memory will not be returned to the pool.
	ManuallyManaged
	// IsOverallocated indicates that the memory for a given tensor is overallocated (i.e. the size-in-use is smaller than the size allocated)
	IsOverallocated
	// IsView indicates that the memory is not owned by the tensor (e.g. when it's sliced)
	IsView
)

// MakeMemoryFlag makes a memory flag. Typical examples:
//
//	akeMemoryFlag(MemoryFlag(0))                             // NativelyAccessible, not manually managed
//	MakeMemoryFlag(NativelyInaccessible, ManuallyManaged)     // NativelyInaccessible,  manually managed memory
func MakeMemoryFlag(fs ...MemoryFlag) (retVal MemoryFlag) {
	if len(fs) == 1 {
		return fs[0]
	}

	for _, f := range fs {
		retVal |= f
	}
	return
}

func (f MemoryFlag) IsNativelyAccessible() bool { return (f & NativelyInaccessible) == 0 }
func (f MemoryFlag) IsManuallyManaged() bool    { return (f & ManuallyManaged) != 0 }
func (f MemoryFlag) IsOverallocated() bool      { return (f & IsOverallocated) != 0 }
func (f MemoryFlag) IsView() bool               { return (f & IsView) != 0 }

// ViewFlag creates a copy of `f`, but the `IsView` field is also set
func (f MemoryFlag) ViewFlag() MemoryFlag { return f | IsView }

func (f MemoryFlag) String() string {
	b := new(strings.Builder)
	if f.IsNativelyAccessible() {
		b.WriteString("NativelyAccessible")
	} else {
		b.WriteString("NativelyInaccessible")
	}
	if f.IsManuallyManaged() {
		b.WriteString("|ManuallyManaged")
	}
	if f.IsOverallocated() {
		b.WriteString("|Overallocated")
	}
	if f.IsView() {
		b.WriteString("|View")
	}
	return b.String()
}

// DataOrder is a flag representing the order of the data
type DataOrder byte

const (
	// ColMajor indicates that the data is stored in a col-major way.
	// A data can only be stored in either ColMajor(1) or RowMajor(0).
	// The way the DataOrder was designed causes the default to be RowMajor
	ColMajor DataOrder = 1 << iota
	// NonContiguous indicates that the data is not contiguous.
	// A data can either be Contiguous (0) or NonContiguous (2).
	// The way DataOrder was designed causes the default to be Contiguous.
	NonContiguous

	// Transposed indicates that the data has been transposed
	Transposed
)

var dataOrderNames = []rune("NonContiguous, RowMajorᵀNonContiguous, ColMajorᵀ")

// MakeDataOrder makes a data order. Typical examples:
//
//	MakeDataOrder(DataOrder(0))            // Row Major, contiguous
//	MakeDataOrder(NonContiguous            // Row Major, non-contiguous
//	MakeDataOrder(ColMajor)                // Col Major, contiguous
//	MakeDataOrder(ColMajor, NonContiguous) // what it says on the tin
func MakeDataOrder(fs ...DataOrder) (retVal DataOrder) {
	if len(fs) == 1 {
		return fs[0]
	}
	for _, f := range fs {
		retVal |= f
	}
	return
}

// IsColMajor returns true if the data order describes a col-major data
func (f DataOrder) IsColMajor() bool { return (f & ColMajor) != 0 }

// IsRowMajor returns true if the data order describes a row-major data
func (f DataOrder) IsRowMajor() bool { return !f.IsColMajor() }

// IsContiguous returns true if the data order describes a contiguous data.
func (f DataOrder) IsContiguous() bool { return !f.IsNotContiguous() }

// IsNotContiguous returns true if the data order describes a noncontiguous data.
func (f DataOrder) IsNotContiguous() bool { return (f & NonContiguous) != 0 }

// IsTransposed returns true if the data order describes whether the data has been tranposed (but not moved)
func (f DataOrder) IsTransposed() bool { return (f & Transposed) != 0 }

// ToggleColMajor toggles the col major mode
func (f DataOrder) ToggleColMajor() DataOrder { return f ^ (ColMajor) }

// ClearTransposed clears any transposed flag.
func (f DataOrder) ClearTransposed() DataOrder { return f &^ (Transposed) }

// HasSameOrder returns true if both data orders are the same (either both are ColMajor or both are RowMajor)
func (f DataOrder) HasSameOrder(other DataOrder) bool {
	return (f.IsColMajor() && other.IsColMajor()) || (f.IsRowMajor() && other.IsRowMajor())
}

func (f DataOrder) String() string {
	var start, end int
	if f.IsRowMajor() {
		end = 23
		if f.IsContiguous() {
			start = 3
		}
	} else {
		end = 47
		start = 24
		if f.IsContiguous() {
			start = 27
		}
	}
	if f.IsTransposed() {
		end++
	}
	return string(dataOrderNames[start:end])
}

// Triangle is a flag representing the "triangle"ness of a matrix
type Triangle byte

const (
	NotTriangle Triangle = iota
	Upper
	Lower
	Symmetric
)
