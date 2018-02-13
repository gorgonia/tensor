package tensor

import "fmt"

func Example_iteratorRowMajor() {
	T := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}))
	it := IteratorFromDense(T)
	fmt.Printf("T:\n%v\n", T)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	//
	// i: 0, coord: [0 1]
	// i: 1, coord: [0 2]
	// i: 2, coord: [1 0]
	// i: 3, coord: [1 1]
	// i: 4, coord: [1 2]
	// i: 5, coord: [0 0]

}

// IteratorFromDense
func Example_iteratorColMajor() {
	T := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}), AsFortran(nil))
	it := IteratorFromDense(T)
	fmt.Printf("T:\n%v\n", T)

	for i, err := it.Start(); err == nil; i, err = it.Next() {
		fmt.Printf("i: %d, coord: %v\n", i, it.Coord())
	}

	// Output:
	// T:
	// ⎡0  2  4⎤
	// ⎣1  3  5⎦
	//
	// i: 0, coord: [0 1]
	// i: 2, coord: [0 2]
	// i: 4, coord: [1 0]
	// i: 1, coord: [1 1]
	// i: 3, coord: [1 2]
	// i: 5, coord: [0 0]

}

// The AsFortran construction option is a bit finnicky.
func Example_asFortran() {
	// Here the data is passed in and directly used without changing the underlying data
	T0 := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}), AsFortran(nil))
	fmt.Printf("T0:\n%vData: %v\n\n", T0, T0.Data())

	// Here the data is passed into the AsFortran construction option, and it assumes that the data is already in
	// row-major form. Therefore a transpose will be performed.
	T1 := New(WithShape(2, 3), AsFortran([]float64{0, 1, 2, 3, 4, 5}))
	fmt.Printf("T1:\n%vData: %v\n\n", T1, T1.Data())

	// Further example of how AsFortran works:
	orig := New(WithShape(2, 3), WithBacking([]float64{0, 1, 2, 3, 4, 5}))
	T2 := New(WithShape(2, 3), AsFortran(orig))
	fmt.Printf("Original\n%vData: %v\n", orig, orig.Data())
	fmt.Printf("T2:\n%vData: %v\n", T2, T2.Data())

	// Output:
	// T0:
	// ⎡0  2  4⎤
	// ⎣1  3  5⎦
	// Data: [0 1 2 3 4 5]
	//
	// T1:
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	// Data: [0 3 1 4 2 5]
	//
	// Original
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	// Data: [0 1 2 3 4 5]
	// T2:
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	// Data: [0 3 1 4 2 5]
}
