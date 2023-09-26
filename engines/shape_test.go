package stdeng

import (
	"testing"

	"gorgonia.org/shapes"
)

var asMatTestCases = []struct {
	input     shapes.Shape
	axis      int
	inclusive bool
	correct   shapes.Shape
}{
	{shapes.Shape{5}, 0, true, shapes.Shape{1, 5}},
	{shapes.Shape{5}, 1, true, shapes.Shape{5, 1}},
	{shapes.Shape{2, 3, 4}, 0, true, shapes.Shape{1, 24}},
	{shapes.Shape{2, 3, 4}, 0, false, shapes.Shape{1, 12}},
	{shapes.Shape{2, 3, 4}, 1, true, shapes.Shape{2, 12}},
	{shapes.Shape{2, 3, 4}, 1, false, shapes.Shape{2, 4}},
	{shapes.Shape{2, 3, 4}, 2, true, shapes.Shape{6, 4}},
	{shapes.Shape{2, 3, 4}, 2, false, shapes.Shape{6, 1}},
}

func TestAsMat(t *testing.T) {
	for _, tc := range asMatTestCases {
		t.Run("", func(t *testing.T) {
			got := asMat(tc.input, tc.axis, tc.inclusive)
			if !got.Eq(tc.correct) {
				t.Errorf("Expected %v, got %v", tc.correct, got)
			}
		})
	}

}
