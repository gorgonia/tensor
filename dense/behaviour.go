package dense

import (
	"reflect"
	"sync"

	"gorgonia.org/shapes"
)

type consBehaviour func(data any, shape shapes.Shape) any

var cblock sync.Mutex
var constructionbehaviour consBehaviour = DefaultConsBehaviour

// UseConstructionBehaviour tells the package to use either of the following
// construction behaviours to construct new *Dense.
//
//	DefaultConsBehaviour
//	APLConsBehaviour
func UseConstructrionBehaviour(b consBehaviour) {
	cblock.Lock()
	constructionbehaviour = b
	cblock.Unlock()
}

func DefaultConsBehaviour(data any, shape shapes.Shape) any {
	panic("Cannot create a *Dense where the length of the backing array is smaller than the desired shape")
}

func APLConsBehaviour(data any, shape shapes.Shape) any {
	d := reflect.ValueOf(data)
	l := d.Len()
	t := d.Type()
	sz := shape.TotalSize()
	retVal := reflect.MakeSlice(t, sz, sz)
	for i := 0; i < retVal.Len(); i++ {
		retVal.Index(i).Set(d.Index(i % l))
	}
	return retVal.Interface()
}
