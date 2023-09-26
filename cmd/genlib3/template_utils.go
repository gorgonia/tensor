package main

import (
	"strings"
	"text/template"
)

var funcs = template.FuncMap{
	"title": strings.Title,
	"lower": strings.ToLower,
}
