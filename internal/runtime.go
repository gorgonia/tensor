package internal

import "runtime"

func ThisFn() string {
	pc, _, _, ok := runtime.Caller(1)
	if !ok {
		return "UNKNOWNFUNC"
	}
	return runtime.FuncForPC(pc).Name()
}
