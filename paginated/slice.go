package paginated

import "gorgonia.org/tensor"

// Slice is used to represent a slice of a paginated
// tensor.
// It represents a starting and ending coordinate along a single dimension of the tensor.
type Slice struct{ S, E int }

// Start is the starting index of a `Slice`
func (i Slice) Start() int { return i.S }

// End is the last index of a `Slice`
func (i Slice) End() int { return i.E }

// Step is set to 1 for a `Slice`
func (i Slice) Step() int { return 1 }

func (i Slice) IsSingleValue() bool {
	return i.S == i.E
}

func (i Slice) intersects(i2 tensor.Slice) bool {
	// i is all larger than i2
	if i.Start() > i2.End() {
		return false
	}

	// i2 is all larger than i
	if i2.Start() > i.End() {
		return false
	}

	return true
}

func (i Slice) contains(point int) bool {
	return point >= i.S && point <= i.E
}
