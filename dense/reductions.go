package dense

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/execution"
)

func (t *Dense[DT]) Sum(opts ...FuncOpt) (retVal *Dense[DT], err error) {
	var fn any
	e := t.e.Workhorse()
	var nativelyAccessible tensor.MemoryFlag
	if !e.WorksWith(nativelyAccessible, t.DataOrder()) {
		fn = "add"
	}
	var z DT
	switch any(z).(type) {
	case float32:
		fn = tensor.ReductionModule[float32]{
			MonotonicReduction: execution.MonotonicSum[float32],
			ReduceFirstN:       execution.Sum0[float32],
			ReduceLastN:        execution.Sum[float32],
			Reduce:             func(a, b float32) float32 { return a + b },
		}
	case float64:
		fn = tensor.ReductionModule[float64]{
			MonotonicReduction: execution.MonotonicSum[float64],
			ReduceFirstN:       execution.Sum0[float64],
			ReduceLastN:        execution.Sum[float64],
			Reduce:             func(a, b float64) float64 { return a + b },
		}
	}

	return t.Reduce(fn, z, opts...)
}
