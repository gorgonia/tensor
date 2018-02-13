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
}
