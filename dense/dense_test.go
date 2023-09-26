package dense

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
	gutils "gorgonia.org/tensor/internal/utils"
)

type ExampleMemory[DT any] []byte

func (m ExampleMemory[DT]) Uintptr() uintptr { return uintptr(unsafe.Pointer(&m[0])) }
func (m ExampleMemory[DT]) MemSize() uintptr { return uintptr(len(m)) }

type ExampleNonStdEng[DT OrderedNum, T tensor.Basic[DT]] struct {
	fake      []DT
	fakeBytes []byte
}

func newExampleNonStdEng[DT OrderedNum, T tensor.Tensor[DT, T]]() *ExampleNonStdEng[DT, T] {
	fake := make([]DT, 64)
	for i := range fake {
		fake[i] = DT(i)
	}
	bytes := gutils.BytesFromSlice[DT](fake)
	return &ExampleNonStdEng[DT, T]{
		fake:      fake,
		fakeBytes: bytes,
	}
}

func (e ExampleNonStdEng[DT, T]) AllocAccessible() bool { return false }

func (e ExampleNonStdEng[DT, T]) Alloc(size int64) (Memory, error) {
	return ExampleMemory[DT](e.fakeBytes), nil
}

func (e ExampleNonStdEng[DT, T]) Free(mem Memory, size int64) error { return nil }

func (e ExampleNonStdEng[DT, T]) Memset(mem Memory, val interface{}) error {
	var z DT
	var ok bool
	if z, ok = val.(DT); !ok {
		return errors.Errorf("Expected val to be of type %T. Got %v of %T instead", z, val, val)
	}
	switch m := mem.(type) {
	case *Dense[DT]:
		for i := range m.data {
			m.data[i] = z
		}
		return nil
	default:
		return errors.Errorf("Unable to memset memory of %T", mem)
	}
	panic("Unreachable")
}

func (e ExampleNonStdEng[DT, T]) Memclr(mem Memory) {
	m := mem.(ExampleMemory[DT])
	for i := range m {
		m[i] = 0
	}
}

func (e ExampleNonStdEng[DT, T]) Memcpy(dst Memory, src Memory) error {
	d := dst.(ExampleMemory[DT])
	s := dst.(ExampleMemory[DT])
	copy(d, s)
	return nil
}

func (e ExampleNonStdEng[DT, T]) Accessible(mem Memory) (Memory, error) { return mem, nil }

func (e ExampleNonStdEng[DT, T]) WorksWith(flag MemoryFlag, order DataOrder) bool {
	return flag&internal.ManuallyManaged > 0
}

func (e ExampleNonStdEng[DT, T]) BasicEng() Engine { return ExampleNonStdEng[DT, tensor.Basic[DT]]{} }

func (e ExampleNonStdEng[DT, T]) MemoryFlag() MemoryFlag { return tensor.NativelyInaccessible }

type consEffect func(assert *assert.Assertions, name string, retVal *Dense[int])

func autoAllocated(assert *assert.Assertions, name string, retVal *Dense[int]) {
	name += ":autoAllocated"
	assert.NotNil(retVal.data, name)
}

func autoShaped(optShape ...int) consEffect {
	return func(assert *assert.Assertions, name string, retVal *Dense[int]) {
		name += ":autoShaped"
		exp := shapes.Shape(optShape)
		assert.Truef(retVal.Shape().Eq(exp), "%v - Expected %v. Got %v", name, exp, retVal.Shape())
	}
}

func hasEngine(optEng ...Engine) consEffect {
	return func(assert *assert.Assertions, name string, retVal *Dense[int]) {
		name += ":hasEngine"
		exp := defaultEngine[int]()
		if len(optEng) > 0 {
			exp = optEng[0]
		}
		assert.Equal(exp, retVal.Engine(), name)
	}
}

func autoColMajor(assert *assert.Assertions, name string, retVal *Dense[int]) {
	name += ":autoColMajor"
	assert.True(retVal.DataOrder().IsColMajor(), name)
}

func backingIs(backing []int) consEffect {
	return func(assert *assert.Assertions, name string, retVal *Dense[int]) {
		name += ":backingIs"
		assert.True(&retVal.data[0] == &backing[0] && len(retVal.data) == len(backing), name)
	}
}

type consCase struct {
	name   string
	opts   []ConsOpt
	panics bool
	effect []consEffect
}

var constructionCases []consCase

func init() {
	backing := []int{1, 2, 3, 4, 5, 6}
	mem := []int{1, 2, 3, 4, 5, 6}
	e := newExampleNonStdEng[int, *Dense[int]]()
	s := int(1337)
	m := ExampleMemory[int](gutils.BytesFromSlice(mem))

	//log.Printf("e.fakeBytes[0]: %p, mem[0]: %p", &e.fakeBytes[0], &mem[0])

	// constructionCase was made with the following python snippet
	// 	co = ["WithShape", "WithBacking", "WithEngine", "FromMemory", "FromScalar", "AsFortran"]
	// 	args = {"WithShape": "2,3", "WithBacking": "backing", "WithEngine": "e", "FromMemory":"m", "FromScalar": "s", "AsFortran":"backing"}
	// 	l = list(chain.from_iterable(combinations(co, r) for r in range(1, len(co)+1)))
	//	print('\n'.join(['{"'+'+'.join([o for o in os])+'",' + "[]ConsOpt{"+", ".join([o+"({})".format(args[o]) for o in os])+"}, false, []consEffect{}}," for os in l]))
	//
	// When adding new construction options, change the last line to:
	// 	print('\n'.join(filter(lambda x: "NEWCONSOPTNAME" in x , ['{"'+'+'.join([o for o in os])+'",' + "[]ConsOpt{"+", ".join([o+"({})".format(args[o]) for o in os])+"}, false, []consEffect{}}," for os in l])))
	constructionCases = []consCase{
		{"WithShape", []ConsOpt{WithShape(2, 3)}, false, []consEffect{autoAllocated}},
		{"WithBacking", []ConsOpt{WithBacking(backing)}, false, []consEffect{autoShaped(6), backingIs(backing)}},
		{"WithEngine", []ConsOpt{WithEngine(e)}, true, nil}, // This panics because there is no size or data, just an engine
		{"FromMemory", []ConsOpt{FromMemory(m)}, false, []consEffect{autoShaped(6), backingIs(mem)}},
		{"FromScalar", []ConsOpt{FromScalar(s)}, false, []consEffect{autoShaped()}},
		{"AsFortran", []ConsOpt{AsFortran(backing)}, false, []consEffect{autoColMajor}},
		{"WithShape+WithBacking", []ConsOpt{WithShape(2, 3), WithBacking(backing)}, false, []consEffect{autoShaped(2, 3), backingIs(backing)}},
		{"WithShape+WithEngine", []ConsOpt{WithShape(2, 3), WithEngine(e)}, false, []consEffect{autoAllocated}},
		{"WithShape+FromMemory", []ConsOpt{WithShape(2, 3), FromMemory(m)}, false, []consEffect{autoShaped(2, 3), backingIs(mem)}},
		{"WithShape+FromScalar", []ConsOpt{WithShape(2, 3), FromScalar(s)}, true, nil}, // you can't set a tensor as a scalar value and set its shape as (2,3)
		{"WithShape+AsFortran", []ConsOpt{WithShape(2, 3), AsFortran(backing)}, false, []consEffect{autoColMajor}},
		{"WithBacking+WithEngine", []ConsOpt{WithBacking(backing), WithEngine(e)}, false, []consEffect{autoShaped(6), backingIs(e.fake)}},
		{"WithBacking+FromMemory", []ConsOpt{WithBacking(backing), FromMemory(m)}, true, nil},                                            // you can't set a backing array then set another backing array.
		{"WithBacking+FromScalar", []ConsOpt{WithBacking(backing), FromScalar(s)}, true, nil},                                            // you can't set a backing array then set a scalar.
		{"WithBacking+AsFortran", []ConsOpt{WithBacking(backing), AsFortran(backing)}, false, []consEffect{autoColMajor, autoShaped(6)}}, // this is OK. because the backing slice is the same. See alternate use of `AsFortran` below
		{"WithEngine+FromMemory", []ConsOpt{WithEngine(e), FromMemory(m)}, false, []consEffect{autoShaped(6), backingIs(e.fake)}},
		{"WithEngine+FromScalar", []ConsOpt{WithEngine(e), FromScalar(s)}, false, []consEffect{autoShaped()}}, // TODO: ???
		{"WithEngine+AsFortran", []ConsOpt{WithEngine(e), AsFortran(backing)}, false, []consEffect{autoColMajor, autoShaped(6), backingIs(e.fake)}},
		{"FromMemory+FromScalar", []ConsOpt{FromMemory(m), FromScalar(s)}, true, nil},
		{"FromMemory+AsFortran", []ConsOpt{FromMemory(m), AsFortran(backing)}, true, nil},
		{"FromScalar+AsFortran", []ConsOpt{FromScalar(s), AsFortran(backing)}, true, nil},

		/*
			{"WithShape+WithBacking+WithEngine", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e)}, false, []consEffect{}},
			{"WithShape+WithBacking+FromMemory", []ConsOpt{WithShape(2, 3), WithBacking(backing), FromMemory(m)}, false, []consEffect{}},
			{"WithShape+WithBacking+FromScalar", []ConsOpt{WithShape(2, 3), WithBacking(backing), FromScalar(s)}, false, []consEffect{}},
			{"WithShape+WithBacking+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithEngine+FromMemory", []ConsOpt{WithShape(2, 3), WithEngine(e), FromMemory(m)}, false, []consEffect{}},
			{"WithShape+WithEngine+FromScalar", []ConsOpt{WithShape(2, 3), WithEngine(e), FromScalar(s)}, false, []consEffect{}},
			{"WithShape+WithEngine+AsFortran", []ConsOpt{WithShape(2, 3), WithEngine(e), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+FromMemory+FromScalar", []ConsOpt{WithShape(2, 3), FromMemory(m), FromScalar(s)}, false, []consEffect{}},
			{"WithShape+FromMemory+AsFortran", []ConsOpt{WithShape(2, 3), FromMemory(m), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithBacking+WithEngine+FromMemory", []ConsOpt{WithBacking(backing), WithEngine(e), FromMemory(m)}, false, []consEffect{}},
			{"WithBacking+WithEngine+FromScalar", []ConsOpt{WithBacking(backing), WithEngine(e), FromScalar(s)}, false, []consEffect{}},
			{"WithBacking+WithEngine+AsFortran", []ConsOpt{WithBacking(backing), WithEngine(e), AsFortran(backing)}, false, []consEffect{}},
			{"WithBacking+FromMemory+FromScalar", []ConsOpt{WithBacking(backing), FromMemory(m), FromScalar(s)}, false, []consEffect{}},
			{"WithBacking+FromMemory+AsFortran", []ConsOpt{WithBacking(backing), FromMemory(m), AsFortran(backing)}, false, []consEffect{}},
			{"WithBacking+FromScalar+AsFortran", []ConsOpt{WithBacking(backing), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithEngine+FromMemory+FromScalar", []ConsOpt{WithEngine(e), FromMemory(m), FromScalar(s)}, false, []consEffect{}},
			{"WithEngine+FromMemory+AsFortran", []ConsOpt{WithEngine(e), FromMemory(m), AsFortran(backing)}, false, []consEffect{}},
			{"WithEngine+FromScalar+AsFortran", []ConsOpt{WithEngine(e), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"FromMemory+FromScalar+AsFortran", []ConsOpt{FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithBacking+WithEngine+FromMemory", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e), FromMemory(m)}, false, []consEffect{}},
			{"WithShape+WithBacking+WithEngine+FromScalar", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e), FromScalar(s)}, false, []consEffect{}},
			{"WithShape+WithBacking+WithEngine+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithBacking+FromMemory+FromScalar", []ConsOpt{WithShape(2, 3), WithBacking(backing), FromMemory(m), FromScalar(s)}, false, []consEffect{}},
			{"WithShape+WithBacking+FromMemory+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), FromMemory(m), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithBacking+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithEngine+FromMemory+FromScalar", []ConsOpt{WithShape(2, 3), WithEngine(e), FromMemory(m), FromScalar(s)}, false, []consEffect{}},
			{"WithShape+WithEngine+FromMemory+AsFortran", []ConsOpt{WithShape(2, 3), WithEngine(e), FromMemory(m), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithEngine+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), WithEngine(e), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+FromMemory+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithBacking+WithEngine+FromMemory+FromScalar", []ConsOpt{WithBacking(backing), WithEngine(e), FromMemory(m), FromScalar(s)}, false, []consEffect{}},
			{"WithBacking+WithEngine+FromMemory+AsFortran", []ConsOpt{WithBacking(backing), WithEngine(e), FromMemory(m), AsFortran(backing)}, false, []consEffect{}},
			{"WithBacking+WithEngine+FromScalar+AsFortran", []ConsOpt{WithBacking(backing), WithEngine(e), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithBacking+FromMemory+FromScalar+AsFortran", []ConsOpt{WithBacking(backing), FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithEngine+FromMemory+FromScalar+AsFortran", []ConsOpt{WithEngine(e), FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithBacking+WithEngine+FromMemory+FromScalar", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e), FromMemory(m), FromScalar(s)}, false, []consEffect{}},
			{"WithShape+WithBacking+WithEngine+FromMemory+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e), FromMemory(m), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithBacking+WithEngine+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithBacking+FromMemory+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithEngine+FromMemory+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), WithEngine(e), FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithBacking+WithEngine+FromMemory+FromScalar+AsFortran", []ConsOpt{WithBacking(backing), WithEngine(e), FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
			{"WithShape+WithBacking+WithEngine+FromMemory+FromScalar+AsFortran", []ConsOpt{WithShape(2, 3), WithBacking(backing), WithEngine(e), FromMemory(m), FromScalar(s), AsFortran(backing)}, false, []consEffect{}},
		*/

		/* Special cases of the above */

		{"AsFortran", []ConsOpt{AsFortran()}, true, nil}, // no backing data at all!
		{"WithShape+WithBacking(Bad Shape)", []ConsOpt{WithShape(3, 3), WithBacking(backing)}, true, nil},
		{"WithShape+FromScalar(Bad Shape)", []ConsOpt{WithShape(2, 3), FromScalar(s)}, true, nil},
		{"WithBacking+AsFortran (Empty AsFortran)", []ConsOpt{WithBacking(backing), AsFortran()}, false, []consEffect{autoColMajor, autoShaped(6)}}, // alternate use for `AsFortran`. This is OK.
		{"WithBacking+AsFortran(Differing backing)", []ConsOpt{WithBacking(backing), AsFortran(mem)}, true, nil},                                    // you can't use a backing array as a row-major, and then set another backing array as col-major while creating a tensor
		{"FromMemory+AsFortran", []ConsOpt{FromMemory(m), AsFortran()}, false, []consEffect{autoColMajor, autoShaped(6), backingIs(e.fake)}},        // alternate use for `AsFortran`. This is OK.
	}
}

func TestNew(t *testing.T) {
	assert := assert.New(t)

	for _, cc := range constructionCases {
		t.Run(cc.name, func(t *testing.T) {
			var T *Dense[int]
			f := func() {
				T = New[int](cc.opts...)
			}
			if cc.panics {
				assert.Panicsf(f, "Expected case %v to panic.", cc.name)
				return
			} else {
				assert.NotPanicsf(f, "Expected case %v to not panic.", cc.name)
				return
			}
			// check on effects

			for _, ef := range cc.effect {
				ef(assert, cc.name, T)
			}

		})
	}
}

func TestDense_Data(t *testing.T) {
	assert := assert.New(t)

	// Standard usecase
	backing := []int{1, 2, 3, 4}
	T := New[int](WithShape(2, 2), WithBacking(backing))
	assert.Equal(backing, T.Data())

	// Scalar, no overallocation
	T = New[int](FromScalar(1337))
	assert.Equal([]int{1337}, T.Data())

	// Scalar, with overallocation
	T = New[int](WithShape(), WithBacking(backing))
	assert.Equal([]int{1}, T.Data())

	// Overallocated
	overallocated := []int{1, 2, 3, 4, 5, 6, 7, 8}
	T = New[int](WithShape(2, 2), WithBacking(overallocated))
	assert.Equal(backing, T.Data())
}

func TestDense_Memset(t *testing.T) {
	assert := assert.New(t)

	// standard, Go accessible tensor
	T := New[string](WithShape(2, 3))
	if err := T.Memset("Hello"); err != nil {
		t.Fatal(err)
	}
	correct := []string{"Hello", "Hello", "Hello", "Hello", "Hello", "Hello"}
	assert.Equal(correct, T.Data())

	// Natively Inaccessible

	T2 := New[int](WithShape(2, 3), WithEngine(newExampleNonStdEng[int, *Dense[int]]()))
	if err := T2.Memset(1337); err != nil {
		t.Fatal(err)
	}
	correctInts := []int{1337, 1337, 1337, 1337, 1337, 1337}
	assert.Equal(correctInts, T2.Data())
}

func TestDense_At(t *testing.T) {
	// basic test
	T := New[int](WithShape(2, 3))
	if err := T.SetAt(1337, 1, 2); err != nil {
		t.Fatalf("Cannot set at T[1,2]. Error: %v", err)
	}

	got, err := T.At(1, 2)
	if err != nil {
		t.Fatalf("Cannot get at T[1,2]. Error: %v", err)
	}
	if got != 1337 {
		t.Fatalf("Expected 1337. Got %v", got)
	}

	// negative addresses are handled but not encouraged
	if err := T.SetAt(1337, -1, 2); err != nil {
		t.Fatalf("Cannot set at T[-1,-2]. Error: %v", err)
	}

	got, err = T.At(-1, 2)
	if err != nil {
		t.Fatalf("Cannot get at T[-1,2]. Error: %v", err)
	}
	if got != 1337 {
		t.Fatalf("Expected 1337. Got %v", got)
	}

	// doing stupid things
	if err := T.SetAt(1337, 10, 10); err == nil {
		t.Error("Expected error when trying to set a value to coordinate [10, 10]. Got none")
	}

	if err := T.SetAt(1337, 0, 0, 2); err == nil {
		t.Error("Expected error when trying to set a value to coordinate [0,0,2]. Got none")
	}

	if _, err := T.At(10, 10); err == nil {
		t.Error("Expected error when trying to get a value to coordinate [10, 10]. Got none")
	}

	if _, err := T.At(0, 0, 2); err == nil {
		t.Error("Expected error when trying to get a value to coordinate [0,0,2]. Got none")
	}

	// natively inaccessible engine
	T = New[int](WithShape(2, 3), WithEngine(newExampleNonStdEng[int, *Dense[int]]()))
	if err := T.SetAt(1337, 1, 2); err == nil {
		t.Fatalf("Expected error. Got none")
	}
	if _, err := T.At(1, 2); err == nil {
		t.Fatalf("Expected error. Got none")
	}

}
