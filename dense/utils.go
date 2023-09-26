package dense

import (
	"fmt"

	"github.com/chewxy/inigo/values/tensor"
	"github.com/chewxy/inigo/values/tensor/internal/errors"
	"github.com/chewxy/inigo/values/tensor/scalar"
	"golang.org/x/exp/constraints"
)

type noopError interface {
	error
	NoOp()
}

func memoryFlagFromEngine(e Engine) (retVal MemoryFlag) {
	if e, ok := e.(NonStandardEngine); ok {
		return e.MemoryFlag()
	}
	return retVal
}

func getEngine[DT any](vs ...Value[DT]) Engine {
	for _, v := range vs {
		if v.Engine() != nil {
			return v.Engine()
		}
	}
	panic("Unreachable")
}

func getAliker[DT any](v Value[DT]) tensor.Aliker[*Dense[DT]] {
	d, ok := v.(tensor.Aliker[*Dense[DT]])
	if ok {
		return d
	}
	return nil
}

// func getNumEngine[DT Num](vs ...NumValue[DT]) NumEngine[DT] {
// 	for _, v := range vs {
// 		if v.Engine() != nil {
// 			return v.Engine().(NumEngine[DT])
// 		}
// 	}
// 	panic("Unreachable")
// }

// getLargestType  gets the largest value type amongst the values provided.
// i.e. if a tensor and a scalar were passed in, then the tensor value will be returned
func getLargestType[DT any](vs ...Value[DT]) Value[DT] {
	var largest Value[DT]
	for _, v := range vs {
		switch v := v.(type) {
		case scalar.Scalar[DT]:
			if largest == nil {
				largest = v
			}
		case tensor.Basic[DT]:
			return v
		}
	}
	return largest

}

// func getLargestNumType[DT Num](vs ...NumValue[DT]) NumValue[DT] {
// 	var largest NumValue[DT]
// 	for _, v := range vs {
// 		switch v := v.(type) {
// 		case Scalar[DT]:
// 			if largest == nil {
// 				largest = v
// 			}
// 		case Basic[DT]:
// 			return v
// 		}
// 	}
// 	return largest
// }

func convert[DST OrderedNum, SRC OrderedNum](data []SRC) []DST {
	retVal := make([]DST, len(data), cap(data))
	for i := range data {
		retVal[i] = DST(data[i])
	}
	return retVal
}

func sliceEq[DT comparable](a, b []DT) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if v != b[i] {
			return false
		}
	}

	return true

}

func max[DT constraints.Ordered](a, b DT) DT {
	if a > b {
		return a
	}
	return b
}

// redFunc2Mod turns a reduction function-like thing into a `ReductionModule`.
//
// Consider reading this function with the `Reduce` method of `*Dense[T]`
func redFunc2Mod[DT any](fn any) (retVal tensor.ReductionModule[DT], err error) {
	switch fn := fn.(type) {
	case func(DT, DT) DT:
		mon := makeMonotonicReduction(fn)
		return tensor.ReductionModule[DT]{
			MonotonicReduction: mon,
			Reduce:             fn,
		}, nil
	case func(DT, DT) (DT, error):
		wrap := mustReduce(fn)
		mon := makeMonotonicReduction(wrap)
		return tensor.ReductionModule[DT]{
			MonotonicReduction: mon,
			ReduceWithErr:      fn,
		}, nil

	case tensor.ReductionModule[DT]:
		if !fn.IsValid() {
			return fn, errors.Errorf("Invalid ReductionModule")
		}
		return fn, nil
	default:
		return retVal, errors.Errorf("Cannot put fn of %T into a ReductionFunctions. Please file a pull request", fn)
	}
}

// makeMonotonicReduction turns a reduction function into a monotonic reduction function
func makeMonotonicReduction[DT any](fn func(DT, DT) DT) func([]DT) DT {
	return func(a []DT) DT {
		if len(a) < 2 {
			panic(fmt.Sprintf("Reduction of an slice of length %d is meaningless", len(a)))
		}
		retVal := a[0]
		for _, v := range a[1:] {
			retVal = fn(retVal, v)
		}
		return retVal
	}
}

func mustReduce[DT any](fn func(a, b DT) (DT, error)) func(DT, DT) DT {
	return func(a, b DT) DT {
		retVal, err := fn(a, b)
		if err != nil {
			panic(err)
		}
		return retVal
	}
}

func isSameSlice[DT any](a, b []DT) bool { return &a[0] == &b[0] }
