package tensor

// ReductionModule is a collection of reduction-related functions.
// At its core, the primary function that a ReductionModule provides is the `Reduce` function, which is a
// function that reduces two values of DT into a single value.
// As such, at the very minimum, the `Reduce` or `ReduceWithErr` field needs to be filled.
//
// The other fields in this struct are alternative reduction functions that a `Reducer` can use to speed up
// reduction.
//
// Lastly, the module also contains flags to indicate the various properties of a reduction function.
type ReductionModule[DT any] struct {
	// Reduce is the basic function to be applied in the reduction
	// This is REQUIRED.
	Reduce func(DT, DT) DT

	/* Alternative reduction functions */

	// ReduceWithErr is an alternative version of Reduce.
	// Either this or Reduce must be filled.
	ReduceWithErr func(DT, DT) (DT, error)

	// When all else fails, resort to using names for a module
	Name string

	/* Supplementary Reduction Functions, which when supplied, will be used to speed things up */

	// MonotonicReduction is a function that reduces a list of DT values into a single DT value.
	// This package will generate a default monotonic reduction function from the Reduce function that
	// can best be characterized as an APL-form of folds.
	MonotonicReduction func([]DT) DT

	// ReduceFirstN is a function that reduces the first axis of a given tensor. The results will be placed
	// in `a`.
	ReduceFirstN func(a, b []DT)

	// ReduceLastN is a standard right-folding reduction function. It is used to reduce the last axis with a
	// default value.
	ReduceLastN func(a []DT, defVal DT) DT

	/* Flags */

	// IsNonCommutative is a flag that indicates if the operation in `Reduce` is non-commutative (e.g. - or /)
	IsNonCommutative bool
}

func (r ReductionModule[DT]) IsValid() bool {
	return r.Reduce != nil || r.ReduceWithErr != nil || r.Name != ""
}
