package array

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestArrayCopy(t *testing.T) {
	assert := assert.New(t)
	a := Make([]float64{1, 2, 3, 4})
	b := Make(make([]float64, 4))
	err := a.MemcpyTo(&b)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(a, b)

	// checking that the regular Go copy() function works correctly
	b.Memset(3.14)
	copy(b.Data(), a.Data())
	assert.Equal(a, b)

	b.Memset(3.14)
	copy(b.DataAsBytes(), a.DataAsBytes())
	assert.Equal(a, b)
}
