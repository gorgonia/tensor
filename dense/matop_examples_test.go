package dense

import "fmt"

func ExampleDense_Concat() {
	a := New[float32](WithShape(2, 3), WithBacking([]float32{
		13, 1, -2,
		-15, -2, 9,
	}))
	b := New[float32](WithShape(2, 3), WithBacking([]float32{
		1, 2, 3,
		4, 5, 6,
	}))

	fmt.Printf("a:\n%v\nb:\n%v\n", a, b)

	c, err := a.Concat(0, b, b, b)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}
	fmt.Printf("concat a to [b,b,b] on axis 1:\n%#v\n", c)

	d, err := a.Concat(1, b, b, b)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	fmt.Printf("concat a to [b,b,b] on axis 1:\n%#v\n", d)

	// Output:
	// a:
	// ⎡ 13    1   -2⎤
	// ⎣-15   -2    9⎦
	//
	// b:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// concat a to [b,b,b] on axis 1:
	// ⎡ 13    1   -2⎤
	// ⎢-15   -2    9⎥
	// ⎢  1    2    3⎥
	// ⎢  4    5    6⎥
	// ⎢  1    2    3⎥
	// ⎢  4    5    6⎥
	// ⎢  1    2    3⎥
	// ⎣  4    5    6⎦
	//
	// concat a to [b,b,b] on axis 1:
	// ⎡ 13    1   -2    1    2    3    1    2    3    1    2    3⎤
	// ⎣-15   -2    9    4    5    6    4    5    6    4    5    6⎦

}
