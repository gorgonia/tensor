package tensor

import (
	"testing"

	"gorgonia.org/shapes"
	"gorgonia.org/tensor/internal"
)

var checkBCShapeCases = []struct {
	name                 string
	a                    shapes.Shape
	b                    shapes.Shape
	newShapeA, newShapeB shapes.Shape
	willErr              bool // not NoOp when calling check
}{
	{"No broadcast necessary", shapes.Shape{2, 2}, shapes.Shape{2, 2}, shapes.Shape{2, 2}, shapes.Shape{2, 2}, false},
	{"Same rank/dims, left operand, outermost", shapes.Shape{1, 2}, shapes.Shape{2, 2}, shapes.Shape{1, 2}, shapes.Shape{2, 2}, false},
	{"Same rank/dims, right operand outermost", shapes.Shape{2, 2}, shapes.Shape{1, 2}, shapes.Shape{2, 2}, shapes.Shape{1, 2}, false},
	{"Same rank/dims, left operand, middle", shapes.Shape{2, 1, 4}, shapes.Shape{2, 3, 4}, shapes.Shape{2, 1, 4}, shapes.Shape{2, 3, 4}, false},
	{"Same rank/dims, right operand, middle", shapes.Shape{2, 3, 4}, shapes.Shape{2, 1, 4}, shapes.Shape{2, 3, 4}, shapes.Shape{2, 1, 4}, false},
	{"Same rank/dims, left operand, innermost", shapes.Shape{2, 3, 1}, shapes.Shape{2, 3, 4}, shapes.Shape{2, 3, 1}, shapes.Shape{2, 3, 4}, false},
	{"Same rank/dims, right operand, innermost", shapes.Shape{2, 3, 4}, shapes.Shape{2, 3, 1}, shapes.Shape{2, 3, 4}, shapes.Shape{2, 3, 1}, false},
	{"Different ranks, left operand", shapes.Shape{2}, shapes.Shape{3, 2}, shapes.Shape{1, 2}, shapes.Shape{3, 2}, true},
}

func TestCheckBroadcastShape(t *testing.T) {
	for _, tc := range checkBCShapeCases {
		t.Run(tc.name, func(t *testing.T) {
			err := shapes.AreBroadcastable(tc.a, tc.b)
			if err := internal.HandleNoOp(err); err != nil && !tc.willErr {
				t.Errorf("%v failed. Error: %v", tc.name, err)
				return
			}

		})
	}
}

func TestCalcBroadcastShapes(t *testing.T) {
	for _, tc := range checkBCShapeCases {
		t.Run(tc.name, func(t *testing.T) {
			apA := &AP{shape: tc.a}
			apB := &AP{shape: tc.b}
			apA.RecalcStrides()
			apB.RecalcStrides()
			newAPA, newAPB := CalcBroadcastShapes(apA, apB)
			if !newAPA.Shape().Eq(tc.newShapeA) {
				t.Errorf("Expected %v. Got %v", tc.newShapeA, newAPA.Shape())
			}

			if !newAPB.Shape().Eq(tc.newShapeB) {
				t.Errorf("Expected %v. Got %v", tc.newShapeB, newAPB.Shape())
			}
			if err := internal.HandleNoOp(shapes.AreBroadcastable(newAPA.Shape(), newAPB.Shape())); err != nil {
				t.Errorf("%v failed. Error: %v", tc.name, err)
				return
			}

		})
	}

}
