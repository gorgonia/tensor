package tensor

import "fmt"

// This example showcases the very basics of the package.
func Example_basics() {
	// Create a (2, 2)-Matrix of integers
	a := New(WithShape(2, 2), WithBacking([]int{1, 2, 3, 4}))
	fmt.Printf("a:\n%v\n", a)

	// Create a (2, 3, 4)-tensor of float32s
	b := New(WithBacking(Range(Float32, 0, 24)), WithShape(2, 3, 4))
	fmt.Printf("b:\n%1.1f", b)

	// Accessing data
	x, _ := b.At(0, 1, 2) // in Numpy syntax: b[0,1,2]
	fmt.Printf("x: %1.1f\n\n", x)

	// Setting data
	b.SetAt(float32(1000), 0, 1, 2)
	fmt.Printf("b:\n%v", b)

	// Output:
	// a:
	// ⎡1  2⎤
	// ⎣3  4⎦
	//
	// b:
	// ⎡ 0.0   1.0   2.0   3.0⎤
	// ⎢ 4.0   5.0   6.0   7.0⎥
	// ⎣ 8.0   9.0  10.0  11.0⎦
	//
	// ⎡12.0  13.0  14.0  15.0⎤
	// ⎢16.0  17.0  18.0  19.0⎥
	// ⎣20.0  21.0  22.0  23.0⎦
	//
	// x: 6.0
	//
	// b:
	// ⎡   0     1     2     3⎤
	// ⎢   4     5  1000     7⎥
	// ⎣   8     9    10    11⎦
	//
	// ⎡  12    13    14    15⎤
	// ⎢  16    17    18    19⎥
	// ⎣  20    21    22    23⎦
}

// This example showcases interactions between different data orders
func Example_differingDataOrders() {
	T0 := New(WithShape(2, 3), WithBacking(Range(Int, 0, 6)))                 // Create a (2, 3)-matrix with the standard row-major backing
	T1 := New(WithShape(2, 3), WithBacking(Range(Int, 0, 6)), AsFortran(nil)) // Create a (2, 3)-matrix with a col-major backing
	T2, _ := Add(T0, T1)
	fmt.Printf("T0:\n%vT1:\n%vT2:\n%vT2 Data Order: %v\n\n", T0, T1, T2, T2.DataOrder())

	// the result's data order is highly dependent on the order of operation. It will take after the first operand
	T0 = New(WithShape(2, 3), WithBacking(Range(Int, 1, 7)), AsFortran(nil)) // Create a (2, 3)-matrix with a col-major backing
	T1 = New(WithShape(2, 3), WithBacking(Range(Int, 1, 7)))                 // Create a (2, 3)-matrix with the standard row-major backing
	T2, _ = Add(T0, T1)
	fmt.Printf("T0:\n%vT1:\n%vT2:\n%vT2 Data Order: %v\n\n", T0, T1, T2, T2.DataOrder())

	reuse := New(WithShape(2, 3), WithBacking([]int{1000, 1000, 1000, 1000, 1000, 1000}))
	fmt.Printf("reuse Data Order: %v\n", reuse.DataOrder())
	T2, _ = Add(T0, T1, WithReuse(reuse))
	fmt.Printf("T2:\n%vT2 Data Order: %v\n\n", T2, T2.DataOrder())

	// Output:
	// 	T0:
	// ⎡0  1  2⎤
	// ⎣3  4  5⎦
	// T1:
	// ⎡0  2  4⎤
	// ⎣1  3  5⎦
	// T2:
	// ⎡ 0   3   6⎤
	// ⎣ 4   7  10⎦
	// T2 Data Order: Contiguous, RowMajor
	//
	//
	// T0:
	// ⎡1  3  5⎤
	// ⎣2  4  6⎦
	// T1:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	// T2:
	// ⎡ 2   5   8⎤
	// ⎣ 6   9  12⎦
	// T2 Data Order: Contiguous, ColMajor
	//
	//
	// reuse Data Order: Contiguous, RowMajor
	// T2:
	// ⎡ 2   5   8⎤
	// ⎣ 6   9  12⎦
	// T2 Data Order: Contiguous, ColMajor

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

// The AsDenseDiag construction option creates a dense diagonal matrix from the input, either a slice or a tensor.
// The resulting shape is automatically inferred from the input vector.
//
// This is like Numpy's `diag()` function, except not stupid. Numpy's `diag()` has been a cause of errors because it's somewhat isometric:
//		>>> np.diag(np.diag(np.array([1,2,3])))
//		array([1,2,3])
func Example_asDenseDiag() {
	T := New(WithShape(3), WithBacking([]int{1, 2, 3}))
	T1 := New(AsDenseDiag(T))
	fmt.Printf("T1:\n%v", T1)

	T2 := New(AsDenseDiag([]float64{3.14, 6.28, 11111}))
	fmt.Printf("T2:\n%v", T2)
	// Output:
	// T1:
	//⎡1  0  0⎤
	//⎢0  2  0⎥
	//⎣0  0  3⎦
	// T2:
	// ⎡ 3.14      0      0⎤
	// ⎢    0   6.28      0⎥
	// ⎣    0      0  11111⎦
}
