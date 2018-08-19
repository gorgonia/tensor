set -ex

go env

go test -v -a -covermode=atomic -coverprofile=test.cover .
go test -tags='avx' -a -covermode=atomic -coverprofile=avx.cover .
go test -tags='sse' -a -covermode=atomic -coverprofile=sse.cover .
go test -tags='inplacetranspose' -a -covermode=atomic -coverprofile=inplacetranspose.cover .
go test -a -covermode=atomic -coverprofile=native.cover ./native/.

# because coveralls only accepts one coverage file at one time... we combine them into one gigantic one
covers=(./test.cover ./avx.cover ./sse.cover ./inplacetranspose.cover ./native.cover)
echo "mode: set" > ./final.cover
tail -q -n +2 "${covers[@]}" >> ./final.cover
goveralls -coverprofile=./final.cover -service=travis-ci

set +ex