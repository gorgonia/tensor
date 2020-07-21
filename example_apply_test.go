package tensor_test

import (
	"fmt"

	"gorgonia.org/tensor"
)

func ExampleDense_Apply() {
	a := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	cube := func(a float64) float64 { return a * a * a }

	b, err := a.Apply(cube)
	if err != nil {
		fmt.Printf("b is an error %v", err)
	}
	fmt.Printf("a and b are the same object - %t\n", a.Eq(b))
	fmt.Printf("a is unmutated\n%v\n", a)

	c, err := a.Apply(cube, tensor.WithReuse(a))
	if err != nil {
		fmt.Printf("c is an error %v\n", err)
	}
	fmt.Printf("a and c are the same object - %t\n", a.Eq(c))

	fmt.Printf("a is now mutated\n%v\n", a)
	// Output:
	// a and b are the same object - false
	// a is unmutated
	// ⎡1  2⎤
	// ⎣3  4⎦
	//
	// a and c are the same object - true
	// a is now mutated
	// ⎡ 1   8⎤
	// ⎣27  64⎦
}
