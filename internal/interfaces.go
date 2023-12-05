package internal

import "gorgonia.org/shapes"

type TODO any

// Memory is a representation of memory of the value.
//
// The main reason for requiring both Uintptr() and Pointer() methods is because while Go currently does not have a compacting
// garbage collector, from the docs of `unsafe`:
//
//	Even if a uintptr holds the address of some object, the garbage collector, will not update that uintptr's value if the object moves,
//	nor will that uintptr keep the object from being reclaimed.
type Memory interface {
	Uintptr() uintptr
	MemSize() uintptr
}

// Engine is a representation of an execution engine.
// While different execution engines can have different capabilities, all execution engines must be able to allocate and free memory
type Engine interface {
	AllocAccessible() bool                           // AllocAccessible returns true if the engine return Go-accessible memory pointers?
	Alloc(size int64) (Memory, error)                // Alloc allocates memory
	Free(mem Memory, size int64) error               // Free rees memory
	Memset(mem Memory, val interface{}) error        // Memset - duh
	Memclr(mem Memory)                               // Memclr - duh
	Memcpy(dst, src Memory) error                    // Memcpy - duh
	Accessible(mem Memory) (Memory, error)           // Accessible returns Go-accesible memory pointers, or errors, if it cannot be done
	WorksWith(flag MemoryFlag, order DataOrder) bool // WorksWith returns true if the memory flag and  data order can be directly worked with

	BasicEng() Engine
	Workhorse() Engine
}

type Iterator interface {
	// Start returns the first index
	Start() (int, error)

	// Next returns the next index. Next is defined as the next value in the coordinates
	// For example: let x be a (5,5) matrix that is row-major. Current index is for the coordinate (3,3).
	// Next() returns the index of (3,4).
	//
	// If there is no underlying data store for (3,4) - say for example, the matrix is a sparse matrix, it return an error.
	// If however, there is an underlying data store for (3,4), but it's not valid (for example, masked tensors), it will not return an error.
	//
	// Second example: let x be a (5,5) matrix that is col-major. Current index is for coordinate (3,3).
	// Next() returns the index of (4,3).
	Next() (int, error)

	// NextValidity is like Next, but returns the validity of the value at the index as well.
	NextValidity() (int, bool, error)

	// NextValid returns the next valid index, as well as a skip count.
	NextValid() (int, int, error)

	// NextInvalid returns the next invalid index, as well as a skip count.
	NextInvalid() (int, int, error)

	// Reset resets the iterator
	Reset()

	// SetReverse tells the iterator to iterate in reverse
	SetReverse()

	// SetForward tells the iterator to iterate forwards
	SetForward()

	// Coord returns the coordinates
	Coord() []int

	// Done returns true when the iterator is done iterating.
	Done() bool

	// Shape returns the shape of the multidimensional tensor it's iterating on.
	Shape() shapes.Shape
}
