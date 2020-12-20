#!/bin/sh

old=$1;
new=$2;

git checkout $old
# https://stackoverflow.com/a/2111099
branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')
echo "Benchmarking $branch (old)"
go test -run=$^ -bench=. > ${branch}.bench
for i in {1..10}
	 do
	 go test -run=$^ -bench=. >> ${branch}.bench
	 done

git checkout $new
branch=$(git symbolic-ref HEAD | sed -e 's,.*/\(.*\),\1,')
echo "Benchmarking $branch (new)"
go test -run=$^ -bench=. > ${branch}.bench
for i in {1..10}
	 do
	 go test -run=$^ -bench=. >> ${branch}.bench
	 done
