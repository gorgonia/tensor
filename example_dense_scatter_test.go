package tensor

import "fmt"

func ExampleScatter() {
	T := New(WithShape(2, 3, 4), WithBacking([]float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,

		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	}))

	indices := New(WithShape(2, 3, 4), WithBacking([]int{
		3, 2, 1, 0,
		3, 2, 1, 0,
		4, 3, 2, 1,

		0, 4, 1, 2,
		4, 4, 4, 4,
		3, 3, 3, 3,
	}))

	s, err := Scatter(T, indices)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("%v\n", s)

	// Output:
	// ⎡ 3   2   1   0   0⎤
	// ⎢ 7   6   5   4   0⎥
	// ⎣ 0  11  10   9   8⎦
	//
	// ⎡ 0   2   3   0   1⎤
	// ⎢ 0   0   0   0   7⎥
	// ⎣ 0   0   0  11   0⎦

}

func ExampleScatter_matrixIndices() {
	T := New(WithShape(2, 3, 4), WithBacking([]float32{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,

		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
	}))

	indices := New(WithShape(5, 4), WithBacking([]int{
		3, 2, 1, 0,
		3, 2, 1, 0,
		4, 3, 2, 1,
		0, 4, 1, 2,
		4, 4, 4, 4,
	}))

	s, err := Scatter(T, indices)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("%v\n", s)

	// Output:
	// ⎡ 3   2   1   0   0⎤
	// ⎢ 7   6   5   4   0⎥
	// ⎢ 0  11  10   9   8⎥
	// ⎢ 0   2   3   0   1⎥
	// ⎣ 0   0   0   0   7⎦

}
