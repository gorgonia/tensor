package sparse

import (
	stdeng "gorgonia.org/tensor/engines"
)

func defaultEngine[DT any]() Engine {
	var e Engine
	var v DT

	switch any(v).(type) { // TODO fill in rest of numeric engine
	case int:
		e = stdeng.StdOrderedNumEngine[int, *CS[int]]{}
	case int8:
		e = stdeng.StdOrderedNumEngine[int8, *CS[int8]]{}
	case int16:
		e = stdeng.StdOrderedNumEngine[int16, *CS[int16]]{}
	case int32:
		e = stdeng.StdOrderedNumEngine[int32, *CS[int32]]{}
	case int64:
		e = stdeng.StdOrderedNumEngine[int64, *CS[int64]]{}
	case uint:
		e = stdeng.StdOrderedNumEngine[uint, *CS[uint]]{}
	case uint8:
		e = stdeng.StdOrderedNumEngine[uint8, *CS[uint8]]{}
	case uint16:
		e = stdeng.StdOrderedNumEngine[uint16, *CS[uint16]]{}
	case uint32:
		e = stdeng.StdOrderedNumEngine[uint32, *CS[uint32]]{}
	case uint64:
		e = stdeng.StdOrderedNumEngine[uint64, *CS[uint64]]{}
	case float32:
		e = stdeng.StdOrderedNumEngine[float32, *CS[float32]]{}
	case float64:
		e = stdeng.StdOrderedNumEngine[float64, *CS[float64]]{}
	default:
		// rv := reflect.ValueOf(v)
		// if rv.Comparable() {
		// 	return stdeng.StdComparableEngine[DT, *Dense[DT]]
		// }
		e = stdeng.StdEng[DT, *CS[DT]]{}
	}
	return e
}
