package tensor

import (
	"fmt"
)

// Data shows how the shape of the *Dense actually affects the return value of .Data().
func ExampleDense_Data() {
	T := New(WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4}))
	fmt.Printf("Basics:\n======\nAny kind of arrays: %v\n", T.Data())

	fmt.Printf("\nScalar-like\n===========\n")
	T = New(WithShape(), FromScalar(3.14))
	fmt.Printf("WithShape(), FromScalar: %v\n", T.Data())

	T = New(WithShape(), WithBacking([]float64{3.14}))
	fmt.Printf("WithShape(), With a slice of 1 as backing: %v\n", T.Data())

	T = New(WithShape(1), FromScalar(3.14))
	fmt.Printf("WithShape(1), With an initial scalar: %v\n", T.Data())

	T = New(WithShape(1, 1), WithBacking([]float64{3.14}))
	fmt.Printf("WithShape(1, 1), With an initial scalar: %v\n", T.Data())

	T = New(WithShape(1, 1), FromScalar(3.14))
	fmt.Printf("WithShape(1, 1), With an initial scalar: %v\n", T.Data())

	T.Reshape()
	fmt.Printf("After reshaping to (): %v\n", T.Data())

	// Output:
	// Basics:
	// ======
	// Any kind of arrays: [1 2 3 4]
	//
	// Scalar-like
	// ===========
	// WithShape(), FromScalar: 3.14
	// WithShape(), With a slice of 1 as backing: 3.14
	// WithShape(1), With an initial scalar: [3.14]
	// WithShape(1, 1), With an initial scalar: [3.14]
	// WithShape(1, 1), With an initial scalar: [3.14]
	// After reshaping to (): 3.14

}
