package dense_test

import (
	"fmt"

	"gorgonia.org/tensor/dense"
)

type int2 struct {
	v int
	s string // string represetation
}

func I(i int) int2 { return int2{v: i} }

func (i int2) Format(f fmt.State, c rune) {
	if i.s == "" {
		fmt.Fprintf(f, "%d", i.v)
		return
	}
	fmt.Fprintf(f, "(%s)", i.s)
}

func add(a, b int2) int2 { return int2{v: a.v + b.v, s: fmt.Sprintf("%s + %s", a, b)} }
func mul(a, b int2) int2 { return int2{v: a.v * b.v, s: fmt.Sprintf("%s × %s", a, b)} }

func ExampleDense_Dot_extension() {
	a := dense.New[int2](dense.WithShape(2, 2), dense.WithBacking([]int2{I(1), I(2), I(3), I(4)}))
	b := dense.New[int2](dense.WithShape(2, 3), dense.WithBacking([]int2{I(6), I(5), I(4), I(3), I(2), I(1)}))
	fmt.Printf("a:\n%v\nb:\n%v\n", a, b)

	c, err := a.Dot(add, mul, b)
	if err != nil {
		fmt.Printf("Err %v\n", err)
		return
	}
	fmt.Printf("a +.× b =\n%v", c)

	// NOTE: notice that the outputs all have a `0 +`...
	// This is because `c` is initialized with 0s. The `add` function is called when writing to `c`.

	// Output:
	// a:
	// ⎡1  2⎤
	// ⎣3  4⎦
	//
	// b:
	// ⎡6  5  4⎤
	// ⎣3  2  1⎦
	//
	// a +.× b =
	// ⎡((0 + (1 × 6)) + (2 × 3))  ((0 + (1 × 5)) + (2 × 2))  ((0 + (1 × 4)) + (2 × 1))⎤
	// ⎣((0 + (3 × 6)) + (4 × 3))  ((0 + (3 × 5)) + (4 × 2))  ((0 + (3 × 4)) + (4 × 1))⎦

}
