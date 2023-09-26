package main

type BinOp struct {
	Name          string
	Symbol        string
	TypeClass     string
	IsCommutative bool
}

var arithOps = []BinOp{
	{"Add", "+", "Addable", true},
	{"Sub", "-", "Num", false},
	{"Mul", "*", "Num", true},
	{"Div", "/", "Num", false},
}

var comparableOps = []BinOp{
	{"ElEq", "==", "comparable", false},
	{"Ne", "!=", "comparable", false},
}

var orderedOps = []BinOp{
	{"Lt", "<", "constraints.Ordered", false},
	{"Lte", "<=", "constraints.Ordered", false},
	{"Gt", ">", "constraints.Ordered", false},
	{"Gte", ">=", "constraints.Ordered", false},
}

var cmpOps = append(orderedOps, comparableOps...)

var cmpOpsNum = []BinOp{
	{"Lt", "<", "OrderedNum", false},
	{"Lte", "<=", "OrderedNum", false},
	{"Gt", ">", "OrderedNum", false},
	{"Gte", ">=", "OrderedNum", false},
	{"ElEq", "==", "Num", false},
	{"Ne", "!=", "Num", false},
}
