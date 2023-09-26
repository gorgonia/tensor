package tensor

import "testing"

func TestIsMonotonicInts(t *testing.T) {
	tests := []struct {
		axes              []int
		expectedMonotonic bool
		expectedIncr1     bool
	}{
		{[]int{0, 1}, true, true},
		{[]int{1, 2, 3}, true, true},
		{[]int{1, 2, 4}, true, false},
		{[]int{3, 2, 1}, false, false},
		{[]int{}, true, false},
	}

	for _, test := range tests {
		monotonic, incr1 := IsMonotonicInts(test.axes)
		if monotonic != test.expectedMonotonic {
			t.Errorf("Expected monotonic to be %v for %v, got %v", test.expectedMonotonic, test.axes, monotonic)
		}
		if incr1 != test.expectedIncr1 {
			t.Errorf("Expected incr1 to be %v for %v, got %v", test.expectedIncr1, test.axes, incr1)
		}
	}
}
