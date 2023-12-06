package stdeng

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/errors"
)

// Gen is a generic engine. It is the most basic engine. To make it useful it needs to be
// specialized with `Inst`.
type Gen struct{}

func (e Gen) AllocAccessible() bool             { return true }
func (e Gen) Alloc(size int64) (Memory, error)  { return nil, errors.NoOp{} }
func (e Gen) Free(mem Memory, size int64) error { return nil }
func (e Gen) Memset(mem Memory, val any) error {
	if ms, ok := mem.(Memsetter); ok {
		return ms.Memset(val)
	}
	return errors.Errorf("Cannot memset %v with StdEng", mem)
}
func (e Gen) Memclr(mem Memory)                     { panic("NYI") }
func (e Gen) Memcpy(dst, src Memory) error          { panic("NYI") }
func (e Gen) Accessible(mem Memory) (Memory, error) { panic("NYI") }
func (e Gen) WorksWith(flags MemoryFlag, order DataOrder) bool {
	return flags.IsNativelyAccessible()
}

func (e Gen) Workhorse() Engine { return e }
func (e Gen) BasicEng() Engine  { return e }

// Inst instantiates a generic Engine in to a specific type .
func Inst[DT any, T tensor.Basic[DT]](gen Gen) Engine {
	panic("XXX")
}
