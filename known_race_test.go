// +build ignore
// +build !race

package tensor

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

// This test will fail the `go test -race`.
//
// This is because FromMemory() will use uintptr in a way that is incorrect according to the checkptr directive of Go 1.14+
//
// Though it's incorrect, it's the only way to use heterogenous, readable memory (i.e. CUDA).
func TestFromMemory(t *testing.T) {
	// dummy memory - this could be an externally malloc'd memory, or a mmap'ed file.
	// but here we're just gonna let Go manage memory.
	s := make([]float64, 100)
	ptr := uintptr(unsafe.Pointer(&s[0]))
	size := uintptr(100 * 8)

	T := New(Of(Float32), WithShape(50, 4), FromMemory(ptr, size))
	if len(T.Float32s()) != 200 {
		t.Error("expected 200 Float32s")
	}
	assert.Equal(t, make([]float32, 200), T.Data())
	assert.True(t, T.IsManuallyManaged(), "Unamanged %v |%v | q: %v", ManuallyManaged, T.flag, (T.flag>>ManuallyManaged)&MemoryFlag(1))

	fail := func() { New(FromMemory(ptr, size), Of(Float32)) }
	assert.Panics(t, fail, "Expected bad New() call to panic")
}
