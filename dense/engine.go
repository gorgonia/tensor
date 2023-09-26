package dense

import (
	stdeng "github.com/chewxy/inigo/values/tensor/engines"
)

func defaultEngine[DT any]() Engine {
	var e Engine
	var v DT

	switch any(v).(type) { // TODO fill in rest of numeric engine
	case int:
		e = stdeng.StdOrderedNumEngine[int, *Dense[int]]{}
	case int8:
		e = stdeng.StdOrderedNumEngine[int8, *Dense[int8]]{}
	case int16:
		e = stdeng.StdOrderedNumEngine[int16, *Dense[int16]]{}
	case int32:
		e = stdeng.StdOrderedNumEngine[int32, *Dense[int32]]{}
	case int64:
		e = stdeng.StdOrderedNumEngine[int64, *Dense[int64]]{}
	case uint8:
		e = stdeng.StdOrderedNumEngine[uint8, *Dense[uint8]]{}
	case uint16:
		e = stdeng.StdOrderedNumEngine[uint16, *Dense[uint16]]{}
	case uint32:
		e = stdeng.StdOrderedNumEngine[uint32, *Dense[uint32]]{}
	case uint64:
		e = stdeng.StdOrderedNumEngine[uint64, *Dense[uint64]]{}
	case float32:
		e = StdFloat32Engine[*Dense[float32]]{}
	case float64:
		e = StdFloat64Engine[*Dense[float64]]{}
	default:
		// rv := reflect.ValueOf(v)
		// if rv.Comparable() {
		// 	return stdeng.StdComparableEngine[DT, *Dense[DT]]
		// }
		e = stdeng.StdEng[DT, *Dense[DT]]{}
	}
	return e
}
