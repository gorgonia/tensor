// Code generated by genlib3. DO NOT EDIT

package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
)

func genAddIden[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(0); err != nil {
			t.Errorf("Memset 0 failed: %v", err)
			return false
		}
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
		if err := b.Memset(0); err != nil {
			t.Errorf("Memset 0 failed: %v", err)
			return false
		}
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
		if err := b.Memset(0); err != nil {
			t.Errorf("Memset 0 failed: %v", err)
			return false
		}
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
		if err := b.Memset(0); err != nil {
			t.Errorf("Memset 0 failed: %v", err)
			return false
		}
		correct := a.Clone()
		incr := b.Clone()
		incr.Zero()

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
		if err := b.Memset(0); err != nil {
			t.Errorf("Memset 0 failed: %v", err)
			return false
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

		// TODO: this check is only because broadcasting doesn't work with tensors that require iterators yet. When
		// it does, this section should be removed.
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.Adder[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Add(b, tensor.AutoBroadcast)
		if err, retEarly := qcErrCheck(t, "Add (Broadcast)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.Equal(correct.Data(), ret.Data())
	}
}
func TestDense_Add(t *testing.T) {
	assert := assert.New(t)

	qcHelper[uint](t, assert, genAddIden[uint])
	qcHelper[uint](t, assert, genAddIdenUnsafe[uint])
	qcHelper[uint](t, assert, genAddIdenReuse[uint])
	qcHelper[uint](t, assert, genAddIdenIncr[uint])
	qcHelper[uint](t, assert, genAddIdenBroadcast[uint])
	qcHelper[uint8](t, assert, genAddIden[uint8])
	qcHelper[uint8](t, assert, genAddIdenUnsafe[uint8])
	qcHelper[uint8](t, assert, genAddIdenReuse[uint8])
	qcHelper[uint8](t, assert, genAddIdenIncr[uint8])
	qcHelper[uint8](t, assert, genAddIdenBroadcast[uint8])
	qcHelper[uint16](t, assert, genAddIden[uint16])
	qcHelper[uint16](t, assert, genAddIdenUnsafe[uint16])
	qcHelper[uint16](t, assert, genAddIdenReuse[uint16])
	qcHelper[uint16](t, assert, genAddIdenIncr[uint16])
	qcHelper[uint16](t, assert, genAddIdenBroadcast[uint16])
	qcHelper[uint32](t, assert, genAddIden[uint32])
	qcHelper[uint32](t, assert, genAddIdenUnsafe[uint32])
	qcHelper[uint32](t, assert, genAddIdenReuse[uint32])
	qcHelper[uint32](t, assert, genAddIdenIncr[uint32])
	qcHelper[uint32](t, assert, genAddIdenBroadcast[uint32])
	qcHelper[uint64](t, assert, genAddIden[uint64])
	qcHelper[uint64](t, assert, genAddIdenUnsafe[uint64])
	qcHelper[uint64](t, assert, genAddIdenReuse[uint64])
	qcHelper[uint64](t, assert, genAddIdenIncr[uint64])
	qcHelper[uint64](t, assert, genAddIdenBroadcast[uint64])
	qcHelper[int](t, assert, genAddIden[int])
	qcHelper[int](t, assert, genAddIdenUnsafe[int])
	qcHelper[int](t, assert, genAddIdenReuse[int])
	qcHelper[int](t, assert, genAddIdenIncr[int])
	qcHelper[int](t, assert, genAddIdenBroadcast[int])
	qcHelper[int8](t, assert, genAddIden[int8])
	qcHelper[int8](t, assert, genAddIdenUnsafe[int8])
	qcHelper[int8](t, assert, genAddIdenReuse[int8])
	qcHelper[int8](t, assert, genAddIdenIncr[int8])
	qcHelper[int8](t, assert, genAddIdenBroadcast[int8])
	qcHelper[int16](t, assert, genAddIden[int16])
	qcHelper[int16](t, assert, genAddIdenUnsafe[int16])
	qcHelper[int16](t, assert, genAddIdenReuse[int16])
	qcHelper[int16](t, assert, genAddIdenIncr[int16])
	qcHelper[int16](t, assert, genAddIdenBroadcast[int16])
	qcHelper[int32](t, assert, genAddIden[int32])
	qcHelper[int32](t, assert, genAddIdenUnsafe[int32])
	qcHelper[int32](t, assert, genAddIdenReuse[int32])
	qcHelper[int32](t, assert, genAddIdenIncr[int32])
	qcHelper[int32](t, assert, genAddIdenBroadcast[int32])
	qcHelper[int64](t, assert, genAddIden[int64])
	qcHelper[int64](t, assert, genAddIdenUnsafe[int64])
	qcHelper[int64](t, assert, genAddIdenReuse[int64])
	qcHelper[int64](t, assert, genAddIdenIncr[int64])
	qcHelper[int64](t, assert, genAddIdenBroadcast[int64])
	qcHelper[float32](t, assert, genAddIden[float32])
	qcHelper[float32](t, assert, genAddIdenUnsafe[float32])
	qcHelper[float32](t, assert, genAddIdenReuse[float32])
	qcHelper[float32](t, assert, genAddIdenIncr[float32])
	qcHelper[float32](t, assert, genAddIdenBroadcast[float32])
	qcHelper[float64](t, assert, genAddIden[float64])
	qcHelper[float64](t, assert, genAddIdenUnsafe[float64])
	qcHelper[float64](t, assert, genAddIdenReuse[float64])
	qcHelper[float64](t, assert, genAddIdenIncr[float64])
	qcHelper[float64](t, assert, genAddIdenBroadcast[float64])

}

func genSubInv[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

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
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

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
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		correct := a.Clone()
		reuse := b.Clone()
		if err := reuse.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

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
			return false
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

		// TODO: this check is only because broadcasting doesn't work with tensors that require iterators yet. When
		// it does, this section should be removed.
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Sub(b, tensor.AutoBroadcast)
		if err, retEarly := qcErrCheck(t, "Sub (Broadcast)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Add(b, tensor.UseUnsafe, tensor.AutoBroadcast)
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.True(allClose(correct.Data(), ret.Data()), "Expected ret to be close to correct.\nCorrect: %v\nGot: %v", correct.Data(), ret.Data())

	}
}
func TestDense_Sub(t *testing.T) {
	assert := assert.New(t)

	qcHelper[uint](t, assert, genSubInv[uint])
	qcHelper[uint](t, assert, genSubInvUnsafe[uint])
	qcHelper[uint](t, assert, genSubInvReuse[uint])
	qcHelper[uint](t, assert, genSubInvBroadcast[uint])
	qcHelper[uint8](t, assert, genSubInv[uint8])
	qcHelper[uint8](t, assert, genSubInvUnsafe[uint8])
	qcHelper[uint8](t, assert, genSubInvReuse[uint8])
	qcHelper[uint8](t, assert, genSubInvBroadcast[uint8])
	qcHelper[uint16](t, assert, genSubInv[uint16])
	qcHelper[uint16](t, assert, genSubInvUnsafe[uint16])
	qcHelper[uint16](t, assert, genSubInvReuse[uint16])
	qcHelper[uint16](t, assert, genSubInvBroadcast[uint16])
	qcHelper[uint32](t, assert, genSubInv[uint32])
	qcHelper[uint32](t, assert, genSubInvUnsafe[uint32])
	qcHelper[uint32](t, assert, genSubInvReuse[uint32])
	qcHelper[uint32](t, assert, genSubInvBroadcast[uint32])
	qcHelper[uint64](t, assert, genSubInv[uint64])
	qcHelper[uint64](t, assert, genSubInvUnsafe[uint64])
	qcHelper[uint64](t, assert, genSubInvReuse[uint64])
	qcHelper[uint64](t, assert, genSubInvBroadcast[uint64])
	qcHelper[int](t, assert, genSubInv[int])
	qcHelper[int](t, assert, genSubInvUnsafe[int])
	qcHelper[int](t, assert, genSubInvReuse[int])
	qcHelper[int](t, assert, genSubInvBroadcast[int])
	qcHelper[int8](t, assert, genSubInv[int8])
	qcHelper[int8](t, assert, genSubInvUnsafe[int8])
	qcHelper[int8](t, assert, genSubInvReuse[int8])
	qcHelper[int8](t, assert, genSubInvBroadcast[int8])
	qcHelper[int16](t, assert, genSubInv[int16])
	qcHelper[int16](t, assert, genSubInvUnsafe[int16])
	qcHelper[int16](t, assert, genSubInvReuse[int16])
	qcHelper[int16](t, assert, genSubInvBroadcast[int16])
	qcHelper[int32](t, assert, genSubInv[int32])
	qcHelper[int32](t, assert, genSubInvUnsafe[int32])
	qcHelper[int32](t, assert, genSubInvReuse[int32])
	qcHelper[int32](t, assert, genSubInvBroadcast[int32])
	qcHelper[int64](t, assert, genSubInv[int64])
	qcHelper[int64](t, assert, genSubInvUnsafe[int64])
	qcHelper[int64](t, assert, genSubInvReuse[int64])
	qcHelper[int64](t, assert, genSubInvBroadcast[int64])
	qcHelper[float32](t, assert, genSubInv[float32])
	qcHelper[float32](t, assert, genSubInvUnsafe[float32])
	qcHelper[float32](t, assert, genSubInvReuse[float32])
	qcHelper[float32](t, assert, genSubInvBroadcast[float32])
	qcHelper[float64](t, assert, genSubInv[float64])
	qcHelper[float64](t, assert, genSubInvUnsafe[float64])
	qcHelper[float64](t, assert, genSubInvReuse[float64])
	qcHelper[float64](t, assert, genSubInvBroadcast[float64])

}

func genMulIden[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Memset 1 failed: %v", err)
			return false
		}
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Mul(b)
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genMulIdenUnsafe[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Memset 1 failed: %v", err)
			return false
		}
		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Mul(b, tensor.UseUnsafe)
		if err, retEarly := qcErrCheck(t, "Mul (Unsafe)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, a) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genMulIdenReuse[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Memset 1 failed: %v", err)
			return false
		}
		correct := a.Clone()
		reuse := b.Clone()
		if err := reuse.Memset(1); err != nil {
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Mul(b, tensor.WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Mul (Reuse)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, reuse) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genMulIdenIncr[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Memset 1 failed: %v", err)
			return false
		}
		correct := a.Clone()
		incr := b.Clone()
		incr.Zero()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Mul(b, tensor.WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Mul (Incr)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.Same(ret, incr) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genMulIdenBroadcast[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a, b *Dense[DT]) bool {
		if err := b.Memset(1); err != nil {
			t.Errorf("Memset 1 failed: %v", err)
			return false
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

		// TODO: this check is only because broadcasting doesn't work with tensors that require iterators yet. When
		// it does, this section should be removed.
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Mul(b, tensor.AutoBroadcast)
		if err, retEarly := qcErrCheck(t, "Mul (Broadcast)", a, b, we, err); retEarly {
			return err == nil
		}
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.Equal(correct.Data(), ret.Data())
	}
}
func TestDense_Mul(t *testing.T) {
	assert := assert.New(t)

	qcHelper[uint](t, assert, genMulIden[uint])
	qcHelper[uint](t, assert, genMulIdenUnsafe[uint])
	qcHelper[uint](t, assert, genMulIdenReuse[uint])
	qcHelper[uint](t, assert, genMulIdenIncr[uint])
	qcHelper[uint](t, assert, genMulIdenBroadcast[uint])
	qcHelper[uint8](t, assert, genMulIden[uint8])
	qcHelper[uint8](t, assert, genMulIdenUnsafe[uint8])
	qcHelper[uint8](t, assert, genMulIdenReuse[uint8])
	qcHelper[uint8](t, assert, genMulIdenIncr[uint8])
	qcHelper[uint8](t, assert, genMulIdenBroadcast[uint8])
	qcHelper[uint16](t, assert, genMulIden[uint16])
	qcHelper[uint16](t, assert, genMulIdenUnsafe[uint16])
	qcHelper[uint16](t, assert, genMulIdenReuse[uint16])
	qcHelper[uint16](t, assert, genMulIdenIncr[uint16])
	qcHelper[uint16](t, assert, genMulIdenBroadcast[uint16])
	qcHelper[uint32](t, assert, genMulIden[uint32])
	qcHelper[uint32](t, assert, genMulIdenUnsafe[uint32])
	qcHelper[uint32](t, assert, genMulIdenReuse[uint32])
	qcHelper[uint32](t, assert, genMulIdenIncr[uint32])
	qcHelper[uint32](t, assert, genMulIdenBroadcast[uint32])
	qcHelper[uint64](t, assert, genMulIden[uint64])
	qcHelper[uint64](t, assert, genMulIdenUnsafe[uint64])
	qcHelper[uint64](t, assert, genMulIdenReuse[uint64])
	qcHelper[uint64](t, assert, genMulIdenIncr[uint64])
	qcHelper[uint64](t, assert, genMulIdenBroadcast[uint64])
	qcHelper[int](t, assert, genMulIden[int])
	qcHelper[int](t, assert, genMulIdenUnsafe[int])
	qcHelper[int](t, assert, genMulIdenReuse[int])
	qcHelper[int](t, assert, genMulIdenIncr[int])
	qcHelper[int](t, assert, genMulIdenBroadcast[int])
	qcHelper[int8](t, assert, genMulIden[int8])
	qcHelper[int8](t, assert, genMulIdenUnsafe[int8])
	qcHelper[int8](t, assert, genMulIdenReuse[int8])
	qcHelper[int8](t, assert, genMulIdenIncr[int8])
	qcHelper[int8](t, assert, genMulIdenBroadcast[int8])
	qcHelper[int16](t, assert, genMulIden[int16])
	qcHelper[int16](t, assert, genMulIdenUnsafe[int16])
	qcHelper[int16](t, assert, genMulIdenReuse[int16])
	qcHelper[int16](t, assert, genMulIdenIncr[int16])
	qcHelper[int16](t, assert, genMulIdenBroadcast[int16])
	qcHelper[int32](t, assert, genMulIden[int32])
	qcHelper[int32](t, assert, genMulIdenUnsafe[int32])
	qcHelper[int32](t, assert, genMulIdenReuse[int32])
	qcHelper[int32](t, assert, genMulIdenIncr[int32])
	qcHelper[int32](t, assert, genMulIdenBroadcast[int32])
	qcHelper[int64](t, assert, genMulIden[int64])
	qcHelper[int64](t, assert, genMulIdenUnsafe[int64])
	qcHelper[int64](t, assert, genMulIdenReuse[int64])
	qcHelper[int64](t, assert, genMulIdenIncr[int64])
	qcHelper[int64](t, assert, genMulIdenBroadcast[int64])
	qcHelper[float32](t, assert, genMulIden[float32])
	qcHelper[float32](t, assert, genMulIdenUnsafe[float32])
	qcHelper[float32](t, assert, genMulIdenReuse[float32])
	qcHelper[float32](t, assert, genMulIdenIncr[float32])
	qcHelper[float32](t, assert, genMulIdenBroadcast[float32])
	qcHelper[float64](t, assert, genMulIden[float64])
	qcHelper[float64](t, assert, genMulIdenUnsafe[float64])
	qcHelper[float64](t, assert, genMulIdenReuse[float64])
	qcHelper[float64](t, assert, genMulIdenIncr[float64])
	qcHelper[float64](t, assert, genMulIdenBroadcast[float64])

}

func genDivInv[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Div(b)
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Mul(b, tensor.UseUnsafe)
		return assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genDivInvUnsafe[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		correct := a.Clone()

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Div(b, tensor.UseUnsafe)
		if err, retEarly := qcErrCheck(t, "Div (Unsafe)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Mul(b, tensor.UseUnsafe)
		return assert.Same(a, ret) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genDivInvReuse[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a *Dense[DT]) bool {
		b := New[DT](WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		correct := a.Clone()
		reuse := b.Clone()
		if err := reuse.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
		}

		var we bool
		if !a.IsNativelyAccessible() {
			we = true
		}
		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Div(b, tensor.WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Div (Reuse)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Mul(b, tensor.UseUnsafe)
		return assert.Same(ret, reuse) &&
			assert.True(correct.Shape().Eq(ret.Shape()), "Expected %v. Got %v", correct.Shape(), ret.Shape()) &&
			assert.Equal(correct.Data(), ret.Data())

	}
}

func genDivInvBroadcast[DT internal.OrderedNum](t *testing.T, assert *assert.Assertions) any {
	return func(a, b *Dense[DT]) bool {
		if err := b.Memset(1); err != nil {
			t.Errorf("Failed to memset: %v", err) // b will always be an accessible engine
			return false
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

		// TODO: this check is only because broadcasting doesn't work with tensors that require iterators yet. When
		// it does, this section should be removed.
		if !a.DataOrder().HasSameOrder(b.DataOrder()) && // iterators required
			a.Shape().TotalSize() != b.Shape().TotalSize() && !(a.Shape().IsScalar() && b.Shape().IsScalar()) { // but not fastpath
			we = true
		}

		_, ok := a.Engine().(tensor.BasicArither[DT, *Dense[DT]])
		we = we || !ok

		ret, err := a.Div(b, tensor.AutoBroadcast)
		if err, retEarly := qcErrCheck(t, "Div (Broadcast)", a, b, we, err); retEarly {
			return err == nil
		}
		ret, err = ret.Mul(b, tensor.UseUnsafe, tensor.AutoBroadcast)
		return assert.True(correct.Shape().Eq(ret.Shape())) &&
			assert.True(allClose(correct.Data(), ret.Data()), "Expected ret to be close to correct.\nCorrect: %v\nGot: %v", correct.Data(), ret.Data())

	}
}
func TestDense_Div(t *testing.T) {
	assert := assert.New(t)

	qcHelper[uint](t, assert, genDivInv[uint])
	qcHelper[uint](t, assert, genDivInvUnsafe[uint])
	qcHelper[uint](t, assert, genDivInvReuse[uint])
	qcHelper[uint](t, assert, genDivInvBroadcast[uint])
	qcHelper[uint8](t, assert, genDivInv[uint8])
	qcHelper[uint8](t, assert, genDivInvUnsafe[uint8])
	qcHelper[uint8](t, assert, genDivInvReuse[uint8])
	qcHelper[uint8](t, assert, genDivInvBroadcast[uint8])
	qcHelper[uint16](t, assert, genDivInv[uint16])
	qcHelper[uint16](t, assert, genDivInvUnsafe[uint16])
	qcHelper[uint16](t, assert, genDivInvReuse[uint16])
	qcHelper[uint16](t, assert, genDivInvBroadcast[uint16])
	qcHelper[uint32](t, assert, genDivInv[uint32])
	qcHelper[uint32](t, assert, genDivInvUnsafe[uint32])
	qcHelper[uint32](t, assert, genDivInvReuse[uint32])
	qcHelper[uint32](t, assert, genDivInvBroadcast[uint32])
	qcHelper[uint64](t, assert, genDivInv[uint64])
	qcHelper[uint64](t, assert, genDivInvUnsafe[uint64])
	qcHelper[uint64](t, assert, genDivInvReuse[uint64])
	qcHelper[uint64](t, assert, genDivInvBroadcast[uint64])
	qcHelper[int](t, assert, genDivInv[int])
	qcHelper[int](t, assert, genDivInvUnsafe[int])
	qcHelper[int](t, assert, genDivInvReuse[int])
	qcHelper[int](t, assert, genDivInvBroadcast[int])
	qcHelper[int8](t, assert, genDivInv[int8])
	qcHelper[int8](t, assert, genDivInvUnsafe[int8])
	qcHelper[int8](t, assert, genDivInvReuse[int8])
	qcHelper[int8](t, assert, genDivInvBroadcast[int8])
	qcHelper[int16](t, assert, genDivInv[int16])
	qcHelper[int16](t, assert, genDivInvUnsafe[int16])
	qcHelper[int16](t, assert, genDivInvReuse[int16])
	qcHelper[int16](t, assert, genDivInvBroadcast[int16])
	qcHelper[int32](t, assert, genDivInv[int32])
	qcHelper[int32](t, assert, genDivInvUnsafe[int32])
	qcHelper[int32](t, assert, genDivInvReuse[int32])
	qcHelper[int32](t, assert, genDivInvBroadcast[int32])
	qcHelper[int64](t, assert, genDivInv[int64])
	qcHelper[int64](t, assert, genDivInvUnsafe[int64])
	qcHelper[int64](t, assert, genDivInvReuse[int64])
	qcHelper[int64](t, assert, genDivInvBroadcast[int64])
	qcHelper[float32](t, assert, genDivInv[float32])
	qcHelper[float32](t, assert, genDivInvUnsafe[float32])
	qcHelper[float32](t, assert, genDivInvReuse[float32])
	qcHelper[float32](t, assert, genDivInvBroadcast[float32])
	qcHelper[float64](t, assert, genDivInv[float64])
	qcHelper[float64](t, assert, genDivInvUnsafe[float64])
	qcHelper[float64](t, assert, genDivInvReuse[float64])
	qcHelper[float64](t, assert, genDivInvBroadcast[float64])

}
