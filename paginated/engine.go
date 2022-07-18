package paginated

import "gorgonia.org/tensor"

/*
  This code is currently only used to satisfy an interface.
  It could possibly be extended with some cool memory management stuff.
*/

// Engine is the default engine for a paginated tensor.
// It is currently only implemented to satisfy the `tensor.Engine` interface.
type Engine struct {
	*Tensor
}

// NewEngine will return a new paginated engine
func NewEngine(p *Tensor) *Engine {
	return &Engine{p}
}

// AllocAccessible returns true if the engine return Go-accessible memory pointers?
func (p *Engine) AllocAccessible() bool {
	return false
}

// Alloc allocates memory
func (p *Engine) Alloc(size int64) (tensor.Memory, error) {
	return p.Tensor, nil
}

// Free frees memory
func (p *Engine) Free(mem tensor.Memory, size int64) error {
	return nil
}

// Memset - duh
func (p *Engine) Memset(mem tensor.Memory, val interface{}) error {
	return nil
}

// Memclr - duh
func (p *Engine) Memclr(mem tensor.Memory) {
	return
}

// Memcpy - duh
func (p *Engine) Memcpy(dst, src tensor.Memory) error {
	return nil
}

// Accessible returns Go-accesible memory pointers, or errors, if it cannot be done
func (p *Engine) Accessible(mem tensor.Memory) (tensor.Memory, error) {
	return p.Tensor, nil
}

// WorksWith returns true if the data order can be directly worked with
func (p *Engine) WorksWith(order tensor.DataOrder) bool {
	return false
}
