package internal

// api_common.go is a file containing all interfaces relating to the final API of the package tensor.
// It is here because Go currently does not support aliasing of generic types

// // Value is a Tensor-like type that only supports reading but not writing data.
// //
// // This allows for scalar values to be used
// type Value[DT any] interface {
// 	Desc
// 	RawAccessor[DT]
// 	Engine() Engine
// }
