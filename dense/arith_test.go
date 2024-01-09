package dense

import (
	"testing"
	"testing/quick"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	gutils "gorgonia.org/tensor/internal/utils"
)

func genAddIden[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.Adder[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Add(b)
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genAddIdenUnsafe[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.Adder[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Add(b, tensor.UseUnsafe)
		if err, retEarly := qcErrCheck(t, "Add (Unsafe)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, a) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genAddIdenReuse[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		correct := a.Clone()
		reuse := a.Clone()
		if err := reuse.Memset(1); err != nil {
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.Adder[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Add(b, tensor.WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Add (Reuse)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, reuse) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func qcHelper[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions, gen func(*testing.T, *assert.Assertions) any) {
	t.Helper()
	conf := &quick.Config{
		Rand:     newRand(),
		MaxCount: quickchecks,
		Values:   qcDense[DT],
	}

	if err := quick.Check(gen(t, assert), conf); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}
}

func TestDense_Add(t *testing.T) {
	assert := assert.New(t)
	qcHelper[float64](t, assert, genAddIden[float64])
	qcHelper[float64](t, assert, genAddIdenUnsafe[float64])
	qcHelper[float64](t, assert, genAddIdenReuse[float64])
}

func TestDense_Add_manual(t *testing.T) {
	assert := assert.New(t)

	// basic
	a := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	c := New[float64](WithShape(2, 3), WithBacking([]float64{2, 4, 6, 8, 10, 12}))
	ret, err := a.Add(b)
	if err != nil {
		t.Errorf("Add failed: %v", err)
	}
	assert.Equal(c.data, ret.data)
	assert.True(c.Shape().Eq(ret.Shape()))
	assert.NotEqual(c.data, a.data)

	// reuse
	reuse := New[float64](WithShape(6), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	ret, err = a.Add(b, WithReuse(reuse))
	assert.Nil(err)
	assert.Equal(ret, reuse)
	assert.Equal(c.data, ret.data)
	assert.True(c.Shape().Eq(ret.Shape()))

	// incr
	c.data = []float64{4, 8, 12, 16, 20, 24}
	ret, err = a.Add(b, WithIncr(reuse))
	assert.Nil(err)
	assert.Equal(c.data, ret.data)
	assert.True(c.Shape().Eq(ret.Shape()))

	// b requires an iterator
	c.data = []float64{2, 4, 6, 11, 13, 15}

	b = New[float64](WithShape(4, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}))
	b, err = b.Slice(SR(0, 4, 2))
	if err != nil {
		t.Fatalf("cannot slice %v", err)
	}
	ret, err = a.Add(b)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	assert.Equal(c.data, ret.data)
	assert.True(c.Shape().Eq(ret.Shape()))

	// incr
	c.data = []float64{6, 12, 18, 27, 33, 39}
	ret, err = a.Add(b, WithIncr(reuse))
	assert.Nil(err)
	assert.Equal(c.data, ret.data)
	assert.True(c.Shape().Eq(ret.Shape()))

}

func TestDense_Add_broadcast(t *testing.T) {
	assert := assert.New(t)
	// broadcast left, inner most
	a := New[float64](WithShape(2, 3, 1), WithBacking(gutils.Range[float64](0, 6)))
	b := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](0, 24)))
	c := New[float64](WithShape(2, 3, 4), WithBacking([]float64{0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27, 28}))
	ret, err := a.Add(b, tensor.AutoBroadcast)
	if err != nil {
		t.Errorf("Add failed: %v", err)
	}
	assert.Equal(c.data, ret.data)
	assert.True(c.Shape().Eq(ret.Shape()))
	assert.NotEqual(c.data, a.data)
}
