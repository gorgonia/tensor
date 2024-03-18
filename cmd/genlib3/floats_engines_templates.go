package main

import "text/template"

const floatUnOpRaw = `func (e {{.EngineName}}[T]) {{.Name}}(ctx context.Context, a, retVal tensor.Basic[{{.Type}}]) (err error){
	{{if eq .PkgFunc "" -}}
	fn := func(x {{.Type}}) {{.Type}} { {{- replace .Body "{{.MathPkg}}" .MathPkg  -}}  }
	{{else -}}
	fn := {{.MathPkg}}.{{.PkgFunc}}
	{{end -}}
	return e.Map(ctx, fn, a, retVal)
}

`

var (
	floatEngUnOp *template.Template
)

func init() {
	floatEngUnOp = template.Must(template.New("float engine unary operations").Funcs(funcs).Parse(floatUnOpRaw))
}
