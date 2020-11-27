package tensor

import (
	"testing"
	"testing/quick"
	"unsafe"
)

type F64 float64

func newF64(f float64) *F64 { r := F64(f); return &r }

func (f *F64) Uintptr() uintptr { return uintptr(unsafe.Pointer(f)) }

func (f *F64) MemSize() uintptr { return 8 }

func (f *F64) Pointer() unsafe.Pointer { return unsafe.Pointer(f) }

func Test_FromMemory(t *testing.T) {
	fn := func(F float64) bool {
		f := newF64(F)
		T := New(WithShape(), Of(Float64), FromMemory(f.Uintptr(), f.MemSize()))
		data := T.Data().(float64)

		if data != F {
			return false
		}
		return true
	}
	if err := quick.Check(fn, nil); err != nil {
		t.Logf("%v", err)
	}
}
