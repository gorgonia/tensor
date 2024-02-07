package dense

import (
	"gorgonia.org/tensor"
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

func TestDense_Ne_manual(t *testing.T) {
	a := New[uint](WithShape(2, 3, 2), WithBacking([]uint{
		36, 6, 0, 35, 0, 92,
		17, 46, 81, 93, 0, 0,
	}))
	b := New[uint](WithShape(2, 3, 2), WithBacking([]uint{
		0, 5, 0, 29, 32, 26,
		56, 0, 41, 0, 16, 48,
	}))
	c := New[uint](WithShape(2, 3, 2), WithBacking([]uint{
		0, 0, 73, 0, 82, 85,
		88, 17, 36, 44, 0, 0,
	}))

	ab, err := a.ElNe(b)
	if err != nil {
		t.Fatal(err)
	}
	bc, err := b.ElNe(c)
	if err != nil {
		t.Fatal(err)
	}
	ac, err := a.ElNe(c)
	if err != nil {
		t.Fatal(err)
	}
	abD := ab.(tensor.Basic[bool]).Data()
	bcD := bc.(tensor.Basic[bool]).Data()
	acD := ac.(tensor.Basic[bool]).Data()
	for i := range abD {
		if abD[i] && bcD[i] && !acD[i] {
			t.Logf("%v\n%v\n%v", a, b, c)
			t.Logf("%v\n%v\n%v", ab, bc, ac)
			t.Fatalf("%dth value is not equal", i)
		}
	}

}
