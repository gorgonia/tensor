package tensor

// type Function[DT any] interface {
// 	func(a, b DT) (DT, error) | func(a DT) (DT, error) | func(a, b DT) DT | func(a DT) DT
// }

// func Perv[DT any, F Function[DT]](function F) func(args ...Value[DT]) (Value[DT], error) {
// 	return func(args ...Value[DT]) (Value[DT], error) {
// 		// TODO check len(args) match the types
// 		switch fn := any(function).(type) {
// 		case func(a, b DT) DT:

// 			x := args[0]
// 			y := args[1]
// 			xs := x.Shape()
// 			ys := y.Shape()

// 			log.Printf("x %T, y %T", x, y)
// 			log.Printf("x.Shape() %v y.Shape() %v", xs, ys)
// 			switch {
// 			case !xs.Eq(shapes.ScalarShape()) && !ys.Eq(shapes.ScalarShape()):
// 			case xs.Eq(shapes.ScalarShape()) && !ys.Eq(shapes.ScalarShape()):
// 				retVal := New[DT](WithShape(ys...))
// 				for i := range retVal.data {
// 					retVal.data[i] = fn(x.Data()[0], y.Data()[i])
// 				}
// 				return retVal, nil
// 			case !xs.Eq(shapes.ScalarShape()) && ys.Eq(shapes.ScalarShape()):
// 				retVal := New[DT](WithShape(xs...))
// 				for i := range retVal.data {
// 					retVal.data[i] = fn(x.Data()[i], y.Data()[0])
// 				}
// 				return retVal, nil
// 			case xs.Eq(shapes.ScalarShape()) && ys.Eq(shapes.ScalarShape()):
// 				return S(fn(x.Data()[0], y.Data()[0])), nil
// 			}
// 		}
// 		panic("Unreachable")
// 	}
// }

// func add_[T Addable](a, b T) T { return a + b }

type Function[DT any] interface {
	func(a, b DT) DT | func(a DT) DT
}

/*
func Perv[DT any, F Function[DT]](fn F) func(args ...Value[DT]) (Value[DT], error) {
	return func(args ...Value[DT]) (Value[DT], error) {
		switch fn := any(fn).(type) {
		case func(a DT) DT:
		case func(a, b DT) DT:
			x := args[0]
			y := args[1]
			xs := x.Shape()
			ys := y.Shape()

			xIsScalar := xs.Eq(shapes.ScalarShape())
			yIsScalar := ys.Eq(shapes.ScalarShape())
			switch {
			case xIsScalar && yIsScalar:
				return S(fn(x.Data()[0], y.Data()[0])), nil
			case !xIsScalar && yIsScalar:
				switch x.Dims() {
				case 0:
					// is a scalar!
				case 1:
					xData := x.Data()
					y := y.Data()[0]
					retVal := New[DT](WithShape(xs...))
					for i := range retVal.data {
						retVal.data[i] = fn(xData[i], y)
					}
					return retVal, nil
				default:
					// pervade deeper!
				}
			case xIsScalar && !yIsScalar:
			case !xIsScalar && !yIsScalar:
				// check dims
			}
		}
		panic("Unreachable")
	}
}
*/
