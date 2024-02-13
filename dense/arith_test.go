package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

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

	// with scalar representation
	c.data = []float64{2, 3, 4, 5, 6, 7}
	b = New[float64](WithShape(), WithBacking([]float64{1}))
	ret, err = a.Add(b)
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
