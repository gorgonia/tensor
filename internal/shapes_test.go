package internal

import "testing"

func TestResolveAxis(t *testing.T) {
	cases := []struct {
		axis   int
		dims   int
		expect int
	}{
		{1, 2, 1},
		{-1, 2, 1},
		{3, 2, 1},
		{-3, 2, 1},
		{1, -2, -1},
		{-1, -2, -1},
	}

	for _, c := range cases {
		got := ResolveAxis(c.axis, c.dims)
		if got != c.expect {
			t.Errorf("Expected %d, got %d for axis %d and dims %d", c.expect, got, c.axis, c.dims)
		}
	}
}
