package tensor

import "fmt"

func Example_sum_Sliced() {
	T := New(WithShape(4, 4), WithBacking([]int{
		1, 2, 3, 4,
		5, 6, 7, 8,
		1, 2, 3, 4,
		5, 6, 7, 8,
	}))
	s, _ := T.Slice(S(1, 3), S(1, 3))
	sum, _ := Sum(s)

	fmt.Printf("T:\n%v\nsliced:\n%v\nSum: %v", T, s, sum)

	// Output:
	// T:
	// ⎡1  2  3  4⎤
	// ⎢5  6  7  8⎥
	// ⎢1  2  3  4⎥
	// ⎣5  6  7  8⎦
	//
	// sliced:
	// ⎡6  7⎤
	// ⎣2  3⎦
	//
	// Sum: 18

}
