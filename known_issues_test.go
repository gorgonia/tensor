package tensor

import (
	"testing"
	"testing/quick"

	"github.com/stretchr/testify/assert"
)

func TestIssue70(t *testing.T) {
	a := 2.0
	b := NewDense(Float64, Shape{1, 1}, WithBacking([]float64{3}))
	var correct interface{} = []float64{6.0}

	res, err := Mul(a, b)
	if err != nil {
		t.Fatalf("Error: %v", err)
	}
	assert.Equal(t, correct, res.Data())
	t.Logf("a %v b %v, res %v", a, b, res)
}

func TestIssue72(t *testing.T) {
	a := New(FromScalar(3.14))
	b := 0.0

	bsa, err := Sub(b, a)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v", bsa)
	ret, err := Sub(b, bsa, UseUnsafe())
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%v %v", ret, bsa)

	invReuseScalar := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		//if !a.Shape().IsScalarEquiv() {
		//	return true
		//}
		b := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, numberTypes, unsignedTypes)
		_, ok := q.Engine().(Suber)
		we = we || !ok
		//log.Printf("b-a(r) | b:%v, a %v, r %v", b, a, reuse)

		ret, err := Sub(b, a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		//log.Printf("b-a(r) | b:%v, a %v, r %v, ret %v", b, a, reuse, ret)
		ret, err = Sub(b, ret, UseUnsafe())

		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}

		return true
	}
	if err := quick.Check(invReuseScalar, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Inv test for Sub (scalar as left, tensor as right) failed: %v", err)
	}
}
