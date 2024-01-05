package dense

import (
	"fmt"

	"gorgonia.org/tensor"
)

func Example_broadcast() {
	A := New[int](WithShape(2, 3, 4), WithBacking([]int{
		1000, 1000, 1000, 1000,
		2000, 2000, 2000, 2000,
		3000, 3000, 3000, 3000,

		-1000, -1000, -1000, -1000,
		-2000, -2000, -2000, -2000,
		-3000, -3000, -3000, -3000,
	}))
	b := New[int](WithShape(4), WithBacking([]int{
		1, -1, 2, -2,
	}))

	C, err := A.Sub(b, tensor.AutoBroadcast)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("A:\n%vB:\n%v\n\nC = A - b:\n%v", A, b, C)

	// Output:
	// A:
	// ⎡ 1000   1000   1000   1000⎤
	// ⎢ 2000   2000   2000   2000⎥
	// ⎣ 3000   3000   3000   3000⎦
	//
	// ⎡-1000  -1000  -1000  -1000⎤
	// ⎢-2000  -2000  -2000  -2000⎥
	// ⎣-3000  -3000  -3000  -3000⎦
	//
	// B:
	// [ 1  -1   2  -2]
	//
	// C = A - b:
	// ⎡  999   1001    998   1002⎤
	// ⎢ 1999   2001   1998   2002⎥
	// ⎣ 2999   3001   2998   3002⎦
	//
	// ⎡-1001   -999  -1002   -998⎤
	// ⎢-2001  -1999  -2002  -1998⎥
	// ⎣-3001  -2999  -3002  -2998⎦
}

func Example_broadcast_nonAuto() {
	A := New[int](WithShape(2, 3, 4), WithBacking([]int{
		1000, 1000, 1000, 1000,
		2000, 2000, 2000, 2000,
		3000, 3000, 3000, 3000,

		-1000, -1000, -1000, -1000,
		-2000, -2000, -2000, -2000,
		-3000, -3000, -3000, -3000,
	}))
	// Note: here the shape has to be manually fixed to be (1,3,1). Automatic broadcasting only
	// works when the broadcasting is on the outside.
	b := New[int](WithShape(1, 3, 1), WithBacking([]int{
		1, 10, -1,
	}))

	C, err := A.Mul(b, tensor.AutoBroadcast)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("A:\n%vB:\n%vC = A × b:\n%v", A, b, C)

	// Output:
	// A:
	// ⎡ 1000   1000   1000   1000⎤
	// ⎢ 2000   2000   2000   2000⎥
	// ⎣ 3000   3000   3000   3000⎦
	//
	// ⎡-1000  -1000  -1000  -1000⎤
	// ⎢-2000  -2000  -2000  -2000⎥
	// ⎣-3000  -3000  -3000  -3000⎦
	//
	// B:
	// ⎡ 1⎤
	// ⎣10⎥
	// ⎢-1⎦
	//
	// C = A × b:
	// ⎡  1000    1000    1000    1000⎤
	// ⎢ 20000   20000   20000   20000⎥
	// ⎣ -3000   -3000   -3000   -3000⎦
	//
	// ⎡ -1000   -1000   -1000   -1000⎤
	// ⎢-20000  -20000  -20000  -20000⎥
	// ⎣  3000    3000    3000    3000⎦
}
