package tensor

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLogSoftMax(t *testing.T) {
	testCases := []struct {
		x              Tensor
		axis           int
		expectedOutput interface{}
	}{
		{
			x: New(
				Of(Float64),
				WithShape(3, 4),
				WithBacking([]float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1}),
			),
			axis:           -1,
			expectedOutput: []float64{-1.5425355294551628, -1.4425355294551627, -1.3425355294551626, -1.2425355294551628, -1.5425355294551628, -1.4425355294551627, -1.3425355294551626, -1.2425355294551628, -1.5425355294551628, -1.4425355294551627, -1.3425355294551629, -1.2425355294551628},
		},
		{
			x: New(
				Of(Float32),
				WithShape(3, 4),
				WithBacking([]float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1}),
			),
			axis:           -1,
			expectedOutput: []float32{-1.5425355294551628, -1.4425355294551627, -1.3425355294551626, -1.2425355294551628, -1.5425355294551628, -1.4425355294551627, -1.3425355294551626, -1.2425355294551628, -1.5425355294551628, -1.4425355294551627, -1.3425355294551629, -1.2425355294551628},
		},
		{
			x: New(
				Of(Float32),
				WithShape(3, 2, 2),
				WithBacking([]float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1}),
			),
			axis:           -1,
			expectedOutput: []float32{-0.7443967, -0.64439666, -0.7443967, -0.64439666, -0.7443967, -0.64439666, -0.7443966, -0.64439666, -0.7443966, -0.64439666, -0.7443967, -0.64439666},
		},
		{
			x: New(
				Of(Float64),
				WithShape(3, 2, 2),
				WithBacking([]float64{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1}),
			),
			axis:           1,
			expectedOutput: []float64{-0.7981388693815918, -0.7981388693815918, -0.5981388693815918, -0.5981388693815919, -0.7981388693815918, -0.7981388693815918, -0.5981388693815919, -0.5981388693815919, -0.7981388693815918, -0.7981388693815918, -0.5981388693815919, -0.5981388693815918},
		},
	}
	for i, tC := range testCases {
		t.Run(fmt.Sprintf("Example #%d - %v %v", i+1, tC.x.Shape(), tC.x.Dtype()), func(t *testing.T) {
			c := assert.New(t)

			output, err := LogSoftMax(tC.x, tC.axis)
			t.Logf("output: %#v", output.Data())

			c.NoError(err)
			c.NotNil(output)

			c.Equal(tC.x.Shape(), output.Shape())
			c.InDeltaSlice(tC.expectedOutput, output.Data(), 1e-6)
		})
	}
}
