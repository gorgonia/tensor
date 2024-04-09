package stdeng_test

import (
	"fmt"

	"gorgonia.org/tensor/dense"
	. "gorgonia.org/tensor/engines"
	gutils "gorgonia.org/tensor/internal/utils"
)

func ExampleStdEng_Stack() {
	d := dense.New[float64](WithShape(2, 3), WithBacking(gutils.Range[float64](1, 7)))
	e := dense.New[float64](WithShape(2, 3), WithBacking(gutils.Range[float64](10, 16)))
	f, err := d.Stack(0, e)
	if err != nil {
		fmt.Printf("Errored %v\n", err)
	}
	fmt.Printf("%v", f)

	// Output:
	// ⎡ 1   2   3⎤
	// ⎣ 4   5   6⎦
	//
	// ⎡10  11  12⎤
	// ⎣13  14  15⎦
}
