package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_NativeVectorF64(t *testing.T) {
	assert := assert.New(t)
	T := New(WithBacking(Range(Float64, 0, 6)), WithShape(6))
	it, err := NativeVectorF64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(6, len(it))
	t.Logf("%v", it)
}

func Test_NativeMatrixF64(t *testing.T) {
	assert := assert.New(t)
	T := New(WithBacking(Range(Float64, 0, 6)), WithShape(2, 3))
	it, err := NativeMatrixF64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	t.Logf("%v", it)
}

func TestNative3TensorF64(t *testing.T) {
	assert := assert.New(t)
	T := New(WithBacking(Range(Float64, 0, 24)), WithShape(2, 3, 4))
	it, err := Native3TensorF64(T)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
	t.Logf("%v", it)
}
