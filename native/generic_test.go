package native_test

import (
	"fmt"

	"gorgonia.org/tensor"
	. "gorgonia.org/tensor/native"
)

type MyType int

func Example_vector() {
	backing := []MyType{
		0, 1, 2, 3,
	}
	T := tensor.New(tensor.WithShape(4), tensor.WithBacking(backing))
	val, err := Vector(T)
	if err != nil {
		fmt.Printf("error: %v", err)
	}
	it := val.([]MyType)
	fmt.Println(it)

	// Output:
	// [0 1 2 3]
}

func Example_matrix() {
	backing := []MyType{
		0, 1,
		2, 3,
		4, 5,
	}
	T := tensor.New(tensor.WithShape(3, 2), tensor.WithBacking(backing))
	val, err := Matrix(T)
	if err != nil {
		fmt.Printf("error: %v", err)
	}

	it := val.([][]MyType)
	fmt.Println(it)

	// Output:
	// [[0 1] [2 3] [4 5]]
}

func Example_tensor3() {
	backing := []MyType{
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,

		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23,
	}
	T := tensor.New(tensor.WithShape(2, 3, 4), tensor.WithBacking(backing))
	val, err := Tensor3(T)
	if err != nil {
		fmt.Printf("error: %v", err)
	}
	it := val.([][][]MyType)
	fmt.Println(it)

	//Output:
	// [[[0 1 2 3] [4 5 6 7] [8 9 10 11]] [[12 13 14 15] [16 17 18 19] [20 21 22 23]]]
}
