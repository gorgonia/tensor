package dense

import (
	"testing"
)

// func TestDense_Lt(t *testing.T) {
// 	assert := assert.New(t)
// 	qcHelper[float64](t, assert, genLtTrans[float64])
// }

func TestDense_Lt_manual(t *testing.T) {
	a := New[uint64](WithShape(4, 3))
	b := New[uint64](WithShape(3))
	c, err := a.Lt(b)
	t.Logf("err %v", err)
	t.Logf("c %v", c)
}
