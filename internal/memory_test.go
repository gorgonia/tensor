package internal

import (
	"testing"
	"testing/quick"
	"unsafe"
)

func TestOverlaps(t *testing.T) {
	overcapOverlapA := make([]int, 4, 8)
	overcapOverlapB := append(overcapOverlapA[2:], 5, 6)

	// This does not overlap
	// This is not guaranteed in all versions of Go and all architectures.
	appendedOverlapA := make([]int, 4)
	appendedOverlapB := append(appendedOverlapA[2:], 5, 6)

	overlapSubsetA := make([]int, 16)
	overlapSubsetB := overlapSubsetA[2:5]

	type args struct {
		a []int
		b []int
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "empty slices",
			args: args{
				a: []int{},
				b: []int{},
			},
			want: false,
		},
		{
			name: "one empty slice",
			args: args{
				a: []int{1, 2, 3},
				b: []int{},
			},
			want: false,
		},
		{
			name: "one nil slice",
			args: args{
				a: nil,
				b: []int{1, 2, 3},
			},
			want: false,
		},
		{
			name: "two nil slices",
			args: args{
				a: nil,
				b: nil,
			},
			want: false,
		},
		{
			name: "overlapping overcapped slices",
			args: args{
				a: overcapOverlapA,
				b: overcapOverlapB,
			},
			want: true,
		},
		{
			name: "B is subset of A",
			args: args{
				a: overlapSubsetA,
				b: overlapSubsetB,
			},
			want: true,
		},
		{
			name: "A is subset of B",
			args: args{
				a: overlapSubsetB,
				b: overlapSubsetA,
			},
			want: true,
		},
		{
			name: "Same slices",
			args: args{
				a: overcapOverlapA,
				b: overcapOverlapA,
			},
			want: true,
		},
		{
			name: "non-overlapping slices",
			args: args{
				a: []int{1, 2, 3},
				b: []int{4, 5, 6},
			},
			want: false,
		},

		// CASES TO WATCH OUT FOR:
		{
			name: "MONITOR: overlapping appended slices",
			args: args{
				a: appendedOverlapA,
				b: appendedOverlapB,
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Overlaps(tt.args.a, tt.args.b); got != tt.want {
				var x int
				fstA, fstB := &tt.args.a[0], &tt.args.b[0]
				aptr, bptr := uintptr(unsafe.Pointer(fstA)), uintptr(unsafe.Pointer(fstB))
				capA, capB := aptr+uintptr(cap(tt.args.a))*unsafe.Sizeof(x), bptr+uintptr(cap(tt.args.b))*unsafe.Sizeof(x)
				t.Errorf("%s overlaps() = %v, want %v. Pointers: %p %p. Caps: 0x%x 0x%x | aptr<bptr: %t, bptr < capA:%t", tt.name, got, tt.want, fstA, fstB, capA, capB, aptr < bptr, bptr < capA)
			}
		})
	}
}

func TestOverlapsQuick(t *testing.T) {
	// Property-based testing using testing/quick
	overlapsWrapper := func(a, b []int) bool {
		var want1, want2 bool
		if len(a) > 0 && len(b) > 0 {
			want1 = (uintptr(unsafe.Pointer(&a[0])) < uintptr(unsafe.Pointer(&b[0])) && uintptr(unsafe.Pointer(&a[0]))+uintptr(cap(a)) > uintptr(unsafe.Pointer(&b[0])))
			want2 = (uintptr(unsafe.Pointer(&a[0])) > uintptr(unsafe.Pointer(&b[0])) && uintptr(unsafe.Pointer(&a[0])) < uintptr(unsafe.Pointer(&b[0]))+uintptr(cap(b)))
		}
		return Overlaps(a, b) == want1 || want2
	}

	if err := quick.Check(overlapsWrapper, nil); err != nil {
		t.Error(err)
	}
}
