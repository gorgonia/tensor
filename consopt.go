package tensor

import (
	"fmt"
	"reflect"

	"gorgonia.org/shapes"
)

type Constructor struct {
	Data      any
	Shape     shapes.Shape
	Engine    Engine
	IsScalar  bool
	AsFortran bool
}

type ConsOpt func(*Constructor)

// IsOK checks some basic rules
func (c *Constructor) IsOK() bool {
	return (c.Data != nil || c.Shape != nil) // at least one of (data, shape) must be not nil, AND
}

func WithBacking(backing any) ConsOpt {
	return func(c *Constructor) {
		if c.Data != nil {
			panic(fmt.Sprintf("WithBacking cannot be called as there is already backing data of %T", c.Data))
		}
		c.Data = backing
	}
}

func WithShape(sh ...int) ConsOpt {
	return func(c *Constructor) {
		if len(sh) == 0 {
			c.Shape = shapes.ScalarShape()
		} else {
			c.Shape = shapes.Shape(sh)
		}
	}
}

func WithEngine(e Engine) ConsOpt {
	return func(c *Constructor) {
		c.Engine = e
	}
}

func FromScalar(v any) ConsOpt {
	return func(c *Constructor) {
		if c.Data != nil {
			panic(fmt.Sprintf("FromScalar cannot be used. A Tensor is to be created with two conflicting sets of backing data, one of %T and one of %T", c.Data, v))
		}
		c.Data = v
		ss := shapes.ScalarShape()
		if !c.Shape.Eq(ss) {
			panic(fmt.Sprintf("FromScalar cannot be used. A Tensor is to be created with two conflicting shapes: %v and %v", c.Shape, ss))
		}
		c.Shape = ss
		c.IsScalar = true
	}
}

func FromMemory(mem Memory) ConsOpt {
	return func(c *Constructor) {
		if c.Data != nil {
			panic(fmt.Sprintf("FromMemory cannot be used. A Tensor is to be created with two conflicting sets of backing data, one of %T and one of %T", c.Data, mem))
		}
		c.Data = mem
	}
}

func AsFortran(data ...any) ConsOpt {
	return func(c *Constructor) {
		if len(data) > 0 {
			if c.Data != nil && !reflectIsSameSlice(c.Data, data[0]) {
				panic(fmt.Sprintf("AsFortran cannot be used. A Tensor is to be created with two conflicting sets of backing data, one of %T and one of %T", c.Data, data[0]))
			}
			c.Data = data[0]
		}

		c.AsFortran = true

	}
}

func reflectIsSameSlice(a, b any) bool {
	va := reflect.ValueOf(a)
	vb := reflect.ValueOf(b)

	// Check if a and b are slices
	if va.Kind() != reflect.Slice || vb.Kind() != reflect.Slice {
		return false
	}

	// Check if a and b are of the same element type
	if va.Type().Elem() != vb.Type().Elem() {
		return false
	}

	// Check if a and b have the same length
	if va.Len() != vb.Len() {
		return false
	}

	if va.Len() == 0 {
		return true
	}

	// Check if a and b point to the same memory location
	if va.Pointer() != vb.Pointer() {
		return false
	}

	return true
}
