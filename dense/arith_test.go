package dense

import (
	"testing"
	"testing/quick"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

/* Quickcheck tests for Add */

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
		reuse := b.Clone()
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

func genAddIdenIncr[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		correct := a.Clone()
		incr := b.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.Adder[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Add(b, tensor.WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Add (Incr)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, incr) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genAddIdenBroadcast[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a, b *Dense[DT]) bool {
		correct := a.Clone()
		correctShape := largestShape(a.Shape(), b.Shape())
		if err := correct.Reshape(correctShape...); err != nil {
			t.Errorf("While reshaping, err: %v", err)
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		// TMP: when iterators are required
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.Adder[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Add(b, tensor.AutoBroadcast)
		if err, retEarly := qcErrCheck(t, "Add (Broadcast)", a, b, we, err); retEarly {
			t.Logf("a.shape %v b.shape %v", a.Shape(), b.Shape())
			return err == nil
		}
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.Equal(correct.Data(), ret.Data())
	}
}

/* Quickcheck tests for Sub */

func genSubInv[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Sub(b)
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Add(b, tensor.UseUnsafe)
		return assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genSubInvUnsafe[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Sub(b, tensor.UseUnsafe)
		if err, retEarly := qcErrCheck(t, "Sub (Unsafe)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Add(b, tensor.UseUnsafe)
		return assert.Same(a, ret) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genSubInvReuse[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		correct := a.Clone()
		reuse := b.Clone()
		reuse.Memset(1)

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Sub(b, tensor.WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Sub (Reuse)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Add(b, tensor.UseUnsafe)
		return assert.Same(ret, reuse) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genSubInvBroadcast[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a, b *Dense[DT]) bool {
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
		}
		correct := a.Clone()
		correctShape := largestShape(a.Shape(), b.Shape())
		if err := correct.Reshape(correctShape...); err != nil {
			t.Errorf("While reshaping, err: %v", err)
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		// TMP: when iterators are required
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Sub(b, tensor.AutoBroadcast)
		if err2, retEarly := qcErrCheck(t, "Sub (Broadcast)", a, b, we, err); retEarly {
			if t.Failed() {
				t.Logf("will err %v. Actually err: %v", we, err)
				t.Logf("ret %v", ret)
				t.Logf("a.shape %v b.shape %v", a.Shape(), b.Shape())
				t.Logf("correct.shape %v ret.shape %v", correct.Shape(), ret.Shape())
				t.Logf("a.DataOrder() %v b.DataOrder() %v", a.DataOrder(), b.DataOrder())
			}
			return err2 == nil
		}
		ret, err = ret.Add(b, tensor.UseUnsafe, tensor.AutoBroadcast)
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.True(allClose(correct.Data(), ret.Data()), "Expected ret to be close to correct.\nCorrect: %v\nGot: %v", correct.Data(), ret.Data())

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
		t.Errorf("%v failed: %v", errors.ThisFn(), err)
	}
}

func TestDense_Add(t *testing.T) {
	assert := assert.New(t)
	qcHelper[float64](t, assert, genAddIden[float64])
	qcHelper[float64](t, assert, genAddIdenUnsafe[float64])
	qcHelper[float64](t, assert, genAddIdenReuse[float64])
	qcHelper[float64](t, assert, genAddIdenIncr[float64])
	qcHelper[float64](t, assert, genAddIdenBroadcast[float64])
}

func TestDense_Sub(t *testing.T) {
	assert := assert.New(t)
	qcHelper[float64](t, assert, genSubInv[float64])
	qcHelper[float64](t, assert, genSubInvUnsafe[float64])
	qcHelper[float64](t, assert, genSubInvReuse[float64])
	qcHelper[float64](t, assert, genSubInvBroadcast[float64])
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

func TestBroadcastDebug(t *testing.T) {
	//assert := assert.New(t)
	// broadcast left, inner most
	a := New[float64](WithShape(5), WithBacking([]float64{1, 2, 3, 4, 5}))
	b := New[float64](WithShape(1), WithBacking([]float64{1}))
	c, err := a.Sub(b, tensor.AutoBroadcast)
	if err != nil {
		t.Logf("err %v", err)
	}
	t.Logf("%v | %v", c, c.Shape())
}
