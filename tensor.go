package tensor

/* API */

// T performs a symbolic transposition and returns a new view.
// func T(t Tensor, axes ...int) (View, error) {
// 	panic("NotYetImplemented")
// }

// R performs a reshape of the tensor and returns a new view.
// func R(t Tensor, shp ...int) (View, error) {
// 	panic("NotYetImplemented")
// }

// Reshape performs a reshape of the tensor and returns a new view.
func Reshape(t Desc, shp ...int) (Desc, error) {
	panic("NotYetImplemented")
}

// Apply applies a function and puts the result in a new
func Apply[T any](t Desc, fn func(T) (T, error)) (Desc, error) {
	panic("NotYetImplemneted")
}

// Reduce applies a reduction function and puts the result in a new
func Reduce[U any](t Desc, fn func(a, b U) U, axis int, defVal U) (Desc, error) {
	panic("NYI")
}

// Scan
func Scan[U any](t Desc, fn func(a, b U) U, axis int) (Desc, error) {
	panic("NYI")
}

/* INTERNAL INTERFACES */
