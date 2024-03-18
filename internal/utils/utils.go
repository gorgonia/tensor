package gutils

import (
	"gorgonia.org/dtype"
)

func GetDatatype[T any]() dtype.Datatype[T] { return dtype.Datatype[T]{} }
