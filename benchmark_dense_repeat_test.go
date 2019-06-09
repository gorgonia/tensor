package tensor

import "testing"

func BenchmarkDenseRepeat(b *testing.B) {
	for _, tst := range repeatTests {
		tst := tst
		b.Run(tst.name, func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				tst.tensor.Repeat(tst.axis, tst.repeats...)
			}
		})
	}
}
