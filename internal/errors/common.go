package errors

const (
	DimMismatch       = "Dimension mismatch. Expected %d, got %d."
	IndexOOBAxis      = "Index %d is out of bounds for axis %d which has size %d."
	InaccessibleData  = "Data in %p inaccessible."
	InvalidSliceIndex = "Invalid slice index. Start: %d, End: %d."
	InvalidAxis       = "Invalid axis %d for ndarray with %d dimensions."
	RepeatedAxis      = "Repeated axis %d in permutation pattern."
	ShapeMismatch     = "Shape mismatch. Expected %v. Got %v."
	ShapeMismatch2    = "Innermost dimension of %v does not match outermost dimension of %v."
	SizeMismatch      = "Array Size Mismatch. Expected %d. Got %d"
	ArrayMismatch     = "Cannot reuse %v. Length of array: %d. Expected length of at least %d."
	DtypeError        = "Expected a tensor of %v. Got %v instead."
	TypeError         = "Expected %T. Got %T instead"

	OpFail = "%v failed."

	EngineIncompatibility = "Engine %T does not work with memory flags %v and data order %v"
	EngineSupport         = "Engine %T does not implement %T, which is needed for %s"

	FailedSanity  = "Failed sanity checks for %s"
	FailedFuncOpt = "Unable to handle FuncOpts for %s"

	NYIPR = "%v is not yet implemented for %v. Please file a pull request to extend the features of Gorgonia"
)
