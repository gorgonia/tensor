package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal"
)

func genLtTrans[DT internal.OrderedNum](t *testing.T, _ *assert.Assertions) any {
	return func(a, b, c *Dense[DT], sameShape bool) bool {
		we := !a.IsNativelyAccessible() || !b.IsNativelyAccessible() || !c.IsNativelyAccessible()

		_, ok1 := a.Engine().(tensor.Ord[DT, *Dense[DT]])
		weAB := we || !ok1 || !a.Shape().Eq(b.Shape())
		ab, err := a.Lt(b)
		if err2, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, weAB, err); retEarly {
			return err2 == nil
		}

		_, ok2 := b.Engine().(tensor.Ord[DT, *Dense[DT]])
		weBC := we || !ok2 || !b.Shape().Eq(c.Shape())
		bc, err := b.Lt(c)
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", b, c, weBC, err); retEarly {
			return err == nil
		}

		ac, err := a.Lt(c)
		weAC := we || !ok1 || !a.Shape().Eq(c.Shape())
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, weAC, err); retEarly {
			return err == nil
		}

		abD := ab.(tensor.Basic[bool]).Data()
		bcD := bc.(tensor.Basic[bool]).Data()
		acD := ac.(tensor.Basic[bool]).Data()
		for i := range abD {
			if abD[i] && bcD[i] && !acD[i] {
				return false
			}
		}
		return true
	}
}

func TestDense_Lt(t *testing.T) {
	assert := assert.New(t)
	qcHelper[float64](t, assert, genLtTrans[float64])
}

func TestDense_Lt_manual(t *testing.T) {
	a := New[float64](WithShape(4, 3))
	b := New[float64](WithShape(3))
	c, err := a.Lt(b)
	t.Logf("err %v", err)
	t.Logf("c %v", c)
}
