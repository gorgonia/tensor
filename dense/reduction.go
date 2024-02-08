package dense

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/internal/execution"
)

func (t *Dense[DT]) Sum(opts ...FuncOpt) (retVal *Dense[DT], err error) {
	module := tensor.ReductionModule[DT]{
		MonotonicReduction: execution.MonotonicSum[DT],
		ReduceFirstN:       execution.Sum0[DT],
		ReduceLastN:        execution.Sum[DT],
		Reduce:             func(a, b DT) DT { return a + b },
	}

	var z DT
	return a.Reduce(module, z, opts...)
}
