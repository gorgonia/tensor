package main

import "text/template"

// level 1 aggregation (internal.E) templates

const (
	eArithRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		case {{reflectKind .}}:
			at := a.{{sliceOf .}}
			bt := b.{{sliceOf .}}
			{{$isDiv := eq $name "Div" -}}
			{{$p := panicsDiv0 . -}}
			switch {
			case as && bs:
				Vec{{$name}}{{short .}}(at, bt)
			case as && !bs:
				{{if and $isDiv $p}} err = {{end}} {{$name}}SV{{short .}}(at[0], bt)
			case !as && bs:
				{{if and $isDiv $p}} err = {{end}} {{$name}}VS{{short .}}(at, bt[0])
			default:
				{{if and $isDiv $p}} err = {{end}} Vec{{$name}}{{short .}}(at, bt)
			}
			return
		{{end -}}
		default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eArithIncrRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	is := isScalar(incr, t)
	if ((as && !bs) || (bs && !as)) && is {
		return errors.Errorf("Cannot increment on scalar increment. a: %d, b %d", a.TypedLen(t), b.TypedLen(t))
	}
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		case {{reflectKind .}}:
			at := a.{{sliceOf .}}
			bt := b.{{sliceOf .}}
			it := incr.{{sliceOf .}}

			switch {
			case as && bs:
				Vec{{$name}}{{short .}}(at, bt)
				if !is {
					return e.Add(t, incr, a)
				}
				it[0]+= at[0]
			case as && !bs:
				{{$name}}IncrSV{{short .}}(at[0], bt, it)
			case !as && bs :
				{{$name}}IncrVS{{short .}}(at, bt[0], it)
			default:
				{{$name}}Incr{{short .}}(at, bt,it)
			}
			return
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`
	eArithIterRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		switch {
		case as && bs :
			Vec{{$name}}{{short .}}(at, bt)
		case as && !bs:
			{{$name}}IterSV{{short .}}(at[0], bt, bit)
		case !as && bs:
			{{$name}}IterVS{{short .}}(at, bt[0], ait)
		default:
			{{$name}}Iter{{short .}}(at, bt, ait, bit)
		}
		return
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}Iter", t)
	}
	`

	eArithIterIncrRaw = `as :=isScalar(a, t)
	bs := isScalar(b, t)
	is := isScalar(incr, t)

	if ((as && !bs) || (bs && !as)) && is {
		return errors.Errorf("Cannot increment on a scalar increment. len(a): %d, len(b) %d", a.TypedLen(t), b.TypedLen(t))
	}
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		it := incr.{{sliceOf .}}
		switch {
		case as && bs:
			Vec{{$name}}{{short .}}(at, bt)
			if !is {
				return e.{{$name}}Iter(t, incr, a, iit, ait)
			}
			it[0] += at[0]
			return
		case as && !bs:
			return {{$name}}IterIncrSV{{short .}}(at[0], bt, it, bit, iit)
		case !as && bs:
			return {{$name}}IterIncrVS{{short .}}(at, bt[0], it, ait, iit)
		default:
			return {{$name}}IterIncr{{short .}}(at, bt, it, ait, bit, iit)
		}
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}IterIncr", t)
	}
	`

	eArithRecvRaw = `as :=isScalar(a, t)
	bs := isScalar(b, t)
	rs := isScalar(recv, t)

	if ((as && !bs) || (bs && !as)) && rs {
		return errors.Errorf("Cannot increment on a scalar increment. len(a): %d, len(b) %d", a.TypedLen(t), b.TypedLen(t))
	}

	{{$name := .Name}}
	switch t{
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		rt := recv.{{sliceOf .}}
		{{$name}}Recv{{short .}}(at, bt, rt)
		return
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}Recv", t)
	}
	`

	eMapRaw = `as := isScalar(a, t)
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		var f0 {{template "fntype0" .}}
		var f1 {{template "fntype1" .}}

		switch f := fn.(type){
		case {{template "fntype0" .}}:
			f0 = f
		case {{template "fntype1" .}}:
			f1 = f
		default:
			return errors.Errorf("Cannot map fn of %T to array", fn)
		}

		at := a.{{sliceOf .}}
		{{if isAddable . -}}
		switch{
		case as && incr && f0 != nil:
			at[0] += f0(at[0])
		case as && incr && f0 == nil:
			var tmp {{asType .}}
			if tmp, err= f1(at[0]); err != nil {
				return
			}
			at[0] += tmp
		case as && !incr && f0 != nil:
			at[0] = f0(at[0])
		case as && !incr && f0 == nil:
			at[0], err = f1(at[0])
		case !as && incr && f0 != nil:
			MapIncr{{short .}}(f0, at)
		case !as && incr && f0 == nil:
			err = MapIncrErr{{short .}}(f1, at)
		case !as && !incr && f0 == nil:
			err = MapErr{{short .}}(f1, at)
		default:
			Map{{short .}}(f0, at)
		}
		{{else -}}
		if incr {
			return errors.Errorf("Cannot perform increment on t of %v", t)
		}
		switch {
		case as && f0 != nil:
			at[0] = f0(at[0])
		case as && f0 == nil:
			at[0], err = f1(at[0])
		case !as && f0 == nil:
			err = MapErr{{short .}}(f1, at)
		default:
			Map{{short .}}(f0, at)
		}
		{{end -}}

		{{end -}}
	default:
		return errors.Errorf("Cannot map t of %v", t)

	}
	`

	eMapIterRaw = `switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		var f0 {{template "fntype0" .}}
		var f1 {{template "fntype1" .}}

		switch f := fn.(type){
		case {{template "fntype0" .}}:
			f0 = f
		case {{template "fntype1" .}}:
			f1 = f
		default:
			return errors.Errorf("Cannot map fn of %T to array", fn)
		}

		{{if isAddable . -}}
		switch {
		case incr && f0 != nil:
			MapIterIncr{{short .}}(f0, at, ait)
		case incr && f0 == nil:
			err = MapIterIncrErr{{short .}}(f1, at, ait)
		case !incr && f0 == nil:
			err = MapIterErr{{short .}}(f1, at, ait)
		default:
			MapIter{{short .}}(f0, at, ait)
		}
		{{else -}}
			if incr {
				return errors.Errorf("Cannot perform increment on t of %v", t)
			}
			switch {
			case f0 == nil:
				err = MapIterErr{{short .}}(f1, at, ait)
			default:
				MapIter{{short .}}(f0, at, ait)
			}
		{{end -}}
		{{end -}}
	default:
			return errors.Errorf("Cannot map t of %v", t)
	}
	`

	eCmpSameRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		{{if isBoolRepr . -}}
		case {{reflectKind .}}:
			at := a.{{sliceOf .}}
			bt := b.{{sliceOf .}}
			switch {
			case as && bs:
				{{$name}}Same{{short .}}(at, bt)
			case as && !bs:
				{{$name}}SameSV{{short .}}(at[0], bt)
			case !as && bs:
				{{$name}}SameVS{{short .}}(at, bt[0])
			default:
				{{$name}}Same{{short .}}(at, bt)
			}
			return
		{{end -}}
		{{end -}}
		default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}`

	eCmpBoolRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	rs := isScalar(retVal, Bool)
	rt := retVal.Bools()

	if ((as && !bs) || (bs && !as)) && rs {
		return errors.Errorf("retVal is a scalar. a: %d, b %d", a.TypedLen(t), b.TypedLen(t))
	}

	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		case {{reflectKind .}}:
			at := a.{{sliceOf .}}
			bt := b.{{sliceOf .}}

			switch {
			case as && bs:
				{{$name}}{{short .}}(at, bt, rt)
			case as && !bs:
				{{$name}}SV{{short .}}(at[0], bt, rt)
			case !as && bs :
				{{$name}}VS{{short .}}(at, bt[0], rt)
			default:
				{{$name}}{{short .}}(at, bt, rt)
			}
			return
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eCmpSameIterRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		{{if isBoolRepr . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		switch {
		case as && bs :
			{{$name}}Same{{short .}}(at, bt)
		case as && !bs:
			{{$name}}SameIterSV{{short .}}(at[0], bt, bit)
		case !as && bs:
			{{$name}}SameIterVS{{short .}}(at, bt[0], ait)
		default:
			{{$name}}SameIter{{short .}}(at, bt, ait, bit)
		}
		return
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eCmpBoolIterRaw = `as :=isScalar(a, t)
	bs := isScalar(b, t)
	rs := isScalar(retVal, Bool)
	rt := retVal.Bools()

	if ((as && !bs) || (bs && !as)) && rs {
		return errors.Errorf("retVal is scalar while len(a): %d, len(b) %d", a.TypedLen(t), b.TypedLen(t))
	}

	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		switch {
		case as && bs:
			{{$name}}{{short .}}(at, bt, rt)
			return
		case as && !bs:
			return {{$name}}IterSV{{short .}}(at[0], bt, rt, bit, rit)
		case !as && bs:
			return {{$name}}IterVS{{short .}}(at, bt[0], rt, ait, rit)
		default:
			return {{$name}}Iter{{short .}}(at, bt, rt, ait, bit, rit)
		}
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`
	eMinMaxSameRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		{{if isOrd . -}}
		case {{reflectKind .}}:
			at := a.{{sliceOf .}}
			bt := b.{{sliceOf .}}
			switch {
			case as && bs:
				Vec{{$name}}{{short .}}(at, bt)
			case as && !bs:
				{{$name}}SV{{short .}}(at[0], bt)
			case !as && bs:
				{{$name}}VS{{short .}}(at, bt[0])
			default:
				Vec{{$name}}{{short .}}(at, bt)
			}
			return
		{{end -}}
		{{end -}}
		default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eMinMaxSameIterRaw = `as := isScalar(a, t)
	bs := isScalar(b, t)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		{{if isOrd . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		switch {
		case as && bs :
			Vec{{$name}}{{short .}}(at, bt)
		case as && !bs:
			{{$name}}IterSV{{short .}}(at[0], bt, bit)
		case !as && bs:
			{{$name}}IterVS{{short .}}(at, bt[0], ait)
		default:
			Vec{{$name}}Iter{{short .}}(at, bt, ait, bit)
		}
		return
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	complexAbs = `{{if eq .Kind.String "complex64" -}}
	{{else if eq .Kind.String "complex128" -}}
	{{end -}}
	`

	eReduceFirstRaw = `{{$name := .Name -}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		dt := data.{{sliceOf .}}
		rt := retVal.{{sliceOf .}}
		switch f := fn.(type){
			case func([]{{asType .}}, []{{asType .}}):
				{{$name | unexport}}{{short .}}(dt, rt, split, size, f)
			case func({{asType .}}, {{asType .}}) {{asType .}}:
				generic{{$name}}{{short .}}(dt, rt, split, size, f)
			default:
				return errors.Errorf(reductionErrMsg, fn)
		}
		return nil
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eReduceLastRaw = `var ok bool
	{{$name := .Name -}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		var def {{asType .}}

		if def, ok = defaultValue.({{asType .}}); !ok {
			return errors.Errorf(defaultValueErrMsg, def, defaultValue, defaultValue)
		}
		dt := data.{{sliceOf .}}
		rt := retVal.{{sliceOf .}}
		switch f := fn.(type){
		case func([]{{asType .}}) {{asType .}}:
			{{$name | unexport}}{{short .}}(dt, rt, dimSize, def, f)
		case func({{asType .}}, {{asType .}}) {{asType .}}:
			generic{{$name}}{{short .}}(dt, rt, dimSize, def, f)
		default:
			return errors.Errorf(reductionErrMsg, fn)
		}
		return nil
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eReduceDefaultRaw = `var ok bool
	{{$name := .Name -}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		var f func({{asType .}}, {{asType .}}) {{asType .}}
		if f, ok = fn.(func({{asType .}}, {{asType .}}) {{asType .}}); !ok {
			return errors.Errorf(reductionErrMsg, fn)
		}
		dt := data.{{sliceOf .}}
		rt := retVal.{{sliceOf .}}
		{{$name | unexport}}{{short .}}(dt, rt, dim0, dimSize, outerStride, stride, expected, f)
		return nil
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eReduceRaw = `var ok bool
	switch t{
		{{range .Kinds -}}
	case {{reflectKind .}}:
		var f func({{asType .}}, {{asType .}}) {{asType .}}
		var def {{asType .}}
		if f, ok = fn.(func({{asType .}}, {{asType .}}) {{asType .}}); !ok {
			return nil, errors.Errorf(reductionErrMsg, fn)
		}
		if def, ok  = defaultValue.({{asType .}}); !ok {
			return nil, errors.Errorf(defaultValueErrMsg, def, defaultValue, defaultValue)
		}
		retVal = Reduce{{short .}}(f, def, a.{{sliceOf .}}...)
		return
		{{end -}}
	default:
		return nil, errors.Errorf("Unsupported type %v for Reduce", t)
	}
	`

	eUnaryRaw = `{{$name := .Name -}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		{{$name}}{{short .}}(a.{{sliceOf .}})
		return nil
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eUnaryIterRaw = `{{$name := .Name -}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		return {{$name}}{{short .}}(a.{{sliceOf .}}, ait)
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eUnaryClampRaw = `var ok bool
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		var min, max {{asType .}}
		if min, ok = minVal.({{asType .}}); !ok {
			return errors.Wrap(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
		}
		if max, ok = maxVal.({{asType .}}); !ok {
			return errors.Wrap(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
		}
		Clamp{{short .}}(a.{{sliceOf .}}, min, max)
		return nil
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for Clamp", t)
	}
	`

	eUnaryClampIterRaw = `var ok bool
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		var min, max {{asType .}}
		if min, ok = minVal.({{asType .}}); !ok {
			return errors.Wrap(errors.Errorf(typeMismatch, min, minVal), "Clamp() min")
		}
		if max, ok = maxVal.({{asType .}}); !ok {
			return errors.Wrap(errors.Errorf(typeMismatch, max, maxVal), "Clamp() max")
		}
		return ClampIter{{short .}}(a.{{sliceOf .}}, ait, min, max)
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for Clamp", t)
	}
	`

	eArgmaxRaw = `var next int
	{{$name := .Name -}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		data := a.{{sliceOf .}}
		tmp := make([]{{asType .}}, 0, lastSize)
		for next,  err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			if len(tmp) == lastSize {
				am := Arg{{$name}}{{short .}}(tmp)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
			}
		}
		if _, ok := err.(NoOpError); ok  {
			err = nil
		}
		return
		{{end -}}
	default:
		return nil, errors.Errorf("Unsupported type %v for Arg{{.Name}}", t)
	}
	`

	eArgmaxMaskedRaw = `newMask := make([]bool, 0, lastSize)
	var next int
	{{$name := .Name -}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		data := a.{{sliceOf .}}
		tmp := make([]{{asType .}}, 0, lastSize)
		for next,  err = it.Next(); err == nil; next, err = it.Next() {
			tmp = append(tmp, data[next])
			newMask = append(newMask, mask[next])
			if len(tmp) == lastSize {
				am := Arg{{$name}}Masked{{short .}}(tmp, mask)
				indices = append(indices, am)

				// reset
				tmp = tmp[:0]
				newMask = newMask[:0]
			}
		}
		if _, ok := err.(NoOpError); ok {
			err = nil
		}
		return
		{{end -}}
	default:
		return nil, errors.Errorf("Unsupported type %v for Arg{{.Name}}", t)
	}
	`

	eArgmaxFlatRaw = `switch t {
	{{$name := .Name -}}
		{{range .Kinds -}}
	case {{reflectKind .}}:
		return Arg{{$name}}{{short .}}(a.{{sliceOf .}})
		{{end -}}
	default:
		return -1
	}
	`

	eArgmaxFlatMaskedRaw = `switch t {
		{{$name := .Name -}}
		{{range .Kinds -}}
	case {{reflectKind .}}:
		return Arg{{$name}}Masked{{short .}}(a.{{sliceOf .}}, mask)
		{{end -}}
	default:
		return -1
	}
	`
)

var (
	eArith         *template.Template
	eArithIncr     *template.Template
	eArithIter     *template.Template
	eArithIterIncr *template.Template
	eArithRecv     *template.Template

	eMap     *template.Template
	eMapIter *template.Template

	eCmpBool     *template.Template
	eCmpSame     *template.Template
	eCmpBoolIter *template.Template
	eCmpSameIter *template.Template

	eMinMaxSame *template.Template
	eMinMaxIter *template.Template

	eReduce        *template.Template
	eReduceFirst   *template.Template
	eReduceLast    *template.Template
	eReduceDefault *template.Template

	eUnary          *template.Template
	eUnaryIter      *template.Template
	eUnaryClamp     *template.Template
	eUnaryClampIter *template.Template

	eArgmax           *template.Template
	eArgmaxMasked     *template.Template
	eArgmaxFlat       *template.Template
	eArgmaxFlatMasked *template.Template
)

func init() {
	eArith = template.Must(template.New("eArith").Funcs(funcs).Parse(eArithRaw))
	eArithIncr = template.Must(template.New("eArithIncr").Funcs(funcs).Parse(eArithIncrRaw))
	eArithIter = template.Must(template.New("eArithIter").Funcs(funcs).Parse(eArithIterRaw))
	eArithIterIncr = template.Must(template.New("eArithIterIncr").Funcs(funcs).Parse(eArithIterIncrRaw))
	eArithRecv = template.Must(template.New("eArithRecv").Funcs(funcs).Parse(eArithRecvRaw))

	eMap = template.Must(template.New("eMap").Funcs(funcs).Parse(eMapRaw))
	eMapIter = template.Must(template.New("eMapIter").Funcs(funcs).Parse(eMapIterRaw))

	eCmpBool = template.Must(template.New("eCmpBool").Funcs(funcs).Parse(eCmpBoolRaw))
	eCmpSame = template.Must(template.New("eCmpSame").Funcs(funcs).Parse(eCmpSameRaw))
	eCmpBoolIter = template.Must(template.New("eCmpBoolIter").Funcs(funcs).Parse(eCmpBoolIterRaw))
	eCmpSameIter = template.Must(template.New("eCmpSameIter").Funcs(funcs).Parse(eCmpSameIterRaw))

	eMinMaxSame = template.Must(template.New("eMinMaxSame").Funcs(funcs).Parse(eMinMaxSameRaw))
	eMinMaxIter = template.Must(template.New("eMinMaxSameIter").Funcs(funcs).Parse(eMinMaxSameIterRaw))

	eReduce = template.Must(template.New("eReduce").Funcs(funcs).Parse(eReduceRaw))
	eReduceFirst = template.Must(template.New("eReduceFirst").Funcs(funcs).Parse(eReduceFirstRaw))
	eReduceLast = template.Must(template.New("eReduceLast").Funcs(funcs).Parse(eReduceLastRaw))
	eReduceDefault = template.Must(template.New("eReduceDefault").Funcs(funcs).Parse(eReduceDefaultRaw))

	eUnary = template.Must(template.New("eUnary").Funcs(funcs).Parse(eUnaryRaw))
	eUnaryIter = template.Must(template.New("eUnaryIter").Funcs(funcs).Parse(eUnaryIterRaw))
	eUnaryClamp = template.Must(template.New("eUnaryClamp").Funcs(funcs).Parse(eUnaryClampRaw))
	eUnaryClampIter = template.Must(template.New("eUnaryClampIter").Funcs(funcs).Parse(eUnaryClampIterRaw))

	eArgmax = template.Must(template.New("argmax").Funcs(funcs).Parse(eArgmaxRaw))
	eArgmaxMasked = template.Must(template.New("argmaxMasked").Funcs(funcs).Parse(eArgmaxMaskedRaw))
	eArgmaxFlat = template.Must(template.New("argmaxFlat").Funcs(funcs).Parse(eArgmaxFlatRaw))
	eArgmaxFlatMasked = template.Must(template.New("argmaxFlatMasked").Funcs(funcs).Parse(eArgmaxFlatMaskedRaw))
}
