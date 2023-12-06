package dense

import (
	"testing"

	"github.com/stretchr/testify/assert"
	gutils "gorgonia.org/tensor/internal/utils"
)

func TestVector(t *testing.T) {
	assert := assert.New(t)
	v := New[float64](WithBacking([]float64{1, 2, 3}))
	it, err := Vector(v)
	assert.NoError(err)
	assert.Equal(3, len(it))
	assert.Equal([]float64{1, 2, 3}, it)
}

func TestMatrix(t *testing.T) {
	assert := assert.New(t)
	m := New[float64](WithShape(2, 3), WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	it, err := Matrix(m)
	assert.NoError(err)
	assert.Equal(2, len(it))
	assert.Equal([]float64{1, 2, 3}, it[0])
	assert.Equal([]float64{4, 5, 6}, it[1])
}

func TestTensor3(t *testing.T) {
	assert := assert.New(t)
	t3 := New[float64](WithShape(2, 3, 4), WithBacking(gutils.Range[float64](1, 2*3*4+1)))
	it, err := Tensor3(t3)
	assert.NoError(err)
	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
	assert.Equal([]float64{1, 2, 3, 4}, it[0][0])
	assert.Equal([]float64{5, 6, 7, 8}, it[0][1])
	assert.Equal([]float64{9, 10, 11, 12}, it[0][2])
	assert.Equal([]float64{13, 14, 15, 16}, it[1][0])
	assert.Equal([]float64{17, 18, 19, 20}, it[1][1])
	assert.Equal([]float64{21, 22, 23, 24}, it[1][2])
}

func TestTensor4(t *testing.T) {
	assert := assert.New(t)
	t4 := New[float64](WithShape(2, 3, 4, 5), WithBacking(gutils.Range[float64](1, 2*3*4*5+1)))
	it, err := Tensor4(t4)
	assert.NoError(err)
	assert.Equal(2, len(it))
	assert.Equal(3, len(it[0]))
	assert.Equal(4, len(it[0][0]))
	assert.Equal(5, len(it[0][0][0]))
	assert.Equal([]float64{1, 2, 3, 4, 5}, it[0][0][0])
	assert.Equal([]float64{6, 7, 8, 9, 10}, it[0][0][1])
	assert.Equal([]float64{11, 12, 13, 14, 15}, it[0][0][2])
	assert.Equal([]float64{16, 17, 18, 19, 20}, it[0][0][3])
	assert.Equal([]float64{21, 22, 23, 24, 25}, it[0][1][0])
	assert.Equal([]float64{26, 27, 28, 29, 30}, it[0][1][1])
	assert.Equal([]float64{31, 32, 33, 34, 35}, it[0][1][2])
	assert.Equal([]float64{36, 37, 38, 39, 40}, it[0][1][3])
	assert.Equal([]float64{41, 42, 43, 44, 45}, it[0][2][0])
	assert.Equal([]float64{46, 47, 48, 49, 50}, it[0][2][1])
	assert.Equal([]float64{51, 52, 53, 54, 55}, it[0][2][2])
	assert.Equal([]float64{56, 57, 58, 59, 60}, it[0][2][3])
	assert.Equal([]float64{61, 62, 63, 64, 65}, it[1][0][0])
	assert.Equal([]float64{66, 67, 68, 69, 70}, it[1][0][1])
	assert.Equal([]float64{71, 72, 73, 74, 75}, it[1][0][2])
	assert.Equal([]float64{76, 77, 78, 79, 80}, it[1][0][3])
	assert.Equal([]float64{81, 82, 83, 84, 85}, it[1][1][0])
	assert.Equal([]float64{86, 87, 88, 89, 90}, it[1][1][1])
	assert.Equal([]float64{91, 92, 93, 94, 95}, it[1][1][2])
	assert.Equal([]float64{96, 97, 98, 99, 100}, it[1][1][3])
	assert.Equal([]float64{101, 102, 103, 104, 105}, it[1][2][0])
	assert.Equal([]float64{106, 107, 108, 109, 110}, it[1][2][1])
	assert.Equal([]float64{111, 112, 113, 114, 115}, it[1][2][2])
	assert.Equal([]float64{116, 117, 118, 119, 120}, it[1][2][3])

}
