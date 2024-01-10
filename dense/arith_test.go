package dense

import (
	"testing"
	"testing/quick"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
	"gorgonia.org/tensor/internal/errors"
)

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

// func TestDense_Add(t *testing.T) {
// 	assert := assert.New(t)
// 	qcHelper[float64](t, assert, genAddIden[float64])
// 	qcHelper[float64](t, assert, genAddIdenUnsafe[float64])
// 	qcHelper[float64](t, assert, genAddIdenReuse[float64])
// 	qcHelper[float64](t, assert, genAddIdenIncr[float64])
// 	qcHelper[float64](t, assert, genAddIdenBroadcast[float64])
// }

// func TestDense_Sub(t *testing.T) {
// 	assert := assert.New(t)
// 	qcHelper[float64](t, assert, genSubInv[float64])
// 	qcHelper[float64](t, assert, genSubInvUnsafe[float64])
// 	qcHelper[float64](t, assert, genSubInvReuse[float64])
// 	qcHelper[float64](t, assert, genSubInvBroadcast[float64])
// }

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
