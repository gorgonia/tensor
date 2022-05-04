package tensor_test

import (
	"fmt"

	"gorgonia.org/tensor"
)

func ExampleTranspose() {
	t := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]int{1, 2, 3, 4, 5, 6}))
	t2, err := tensor.Transpose(t)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}
	fmt.Printf("Transpose is a safe operation.\nT:\n%v\nT':\n%v\n", t, t2)
	fmt.Printf("The data is changed:\nT : %v\nT': %v", t.Data(), t2.Data())

	// Output:
	// Transpose is a safe operation.
	// T:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// T':
	// ⎡1  4⎤
	// ⎢2  5⎥
	// ⎣3  6⎦
	//
	// The data is changed:
	// T : [1 2 3 4 5 6]
	// T': [1 4 2 5 3 6]

}

func ExampleT() {
	t := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]int{1, 2, 3, 4, 5, 6}))
	t2, err := tensor.T(t)
	if err != nil {
		fmt.Printf("ERR: %v\n", err)
	}
	fmt.Printf("T is a safe version of the .T() method\nT:\n%v\nT':\n%v\n", t, t2)
	fmt.Printf("The data is unchanged:\nT : %v\nT': %v\n", t.Data(), t2.Data())

	// Output:
	// T is a safe version of the .T() method
	// T:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// T':
	// ⎡1  4⎤
	// ⎢2  5⎥
	// ⎣3  6⎦
	//
	// The data is unchanged:
	// T : [1 2 3 4 5 6]
	// T': [1 2 3 4 5 6]

}
