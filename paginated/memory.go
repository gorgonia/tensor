package paginated

import (
	"unsafe"

	"gorgonia.org/tensor"
)

// Engine ...
func (p *Tensor) Engine() tensor.Engine {
	return p.engine
}

// MemSize returns the current size of the in-memory buffer.
func (p *Tensor) MemSize() uintptr {
	return uintptr(p.cache.Len() * p.pageSize)
}

// Pointer returns a pointer to the first value
// in the in-memory buffer.
func (p *Tensor) Pointer() unsafe.Pointer {
	return nil
}

// IsNativelyAccessible returns `false` for paginated
// tensors.
func (p *Tensor) IsNativelyAccessible() bool {
	return false
}

// IsManuallyManaged returns false for paginated
// tensors.
func (p *Tensor) IsManuallyManaged() bool {
	return false
}

// Uintptr is the same as `Pointer()` but returns
// a `uintptr` rather than an `unsafe.Pointer`
func (p *Tensor) Uintptr() uintptr {
	return uintptr(0)
}
