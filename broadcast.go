package tensor

// BinOp is a function that takes two Tensor and returns a (Tensor, error)
type BinOp func(a, b Tensor, opts ...FuncOpt) (Tensor, error)

// DenseBinOp is a function that takes two *Dense and returns a (*Dense, error)
type DenseBinOp func(a, b *Dense, opts ...FuncOpt) (*Dense, error)

// Broadcast is an operation to perform a binary operation with the provided broadcast patterns.
//
// If both broadcast patterns are nil, then magic happens - the broadcast axes is inferred automatically.
// This is considered to be bad practice
func Broadcast(op BinOp, onLeft, onRight []Axis) BinOp {
	return func(a, b Tensor, opts ...FuncOpt) (Tensor, error) { panic("NYI") }
}

// BroadcastD is an operation to perform a binary operation with the provided broadcast patterns
//
// If both broadcast patterns are nil, then magic happens - the broadcast axes is inferred automatically.
// This is considered to be bad practice
func BroadcastD(op DenseBinOp, onLeft, onRight []Axis) DenseBinOp {
	return func(a, b Tensor, opts ...FuncOpt) (Tensor, error) { panic("NYI") }
}
