package tensor

func IsMatrix(t Desc) bool { return t.Dims() == 2 }
