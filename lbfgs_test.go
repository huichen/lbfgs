package lbfgs

import (
	"fmt"
	"testing"
)

func TestLBFGS(t *testing.T) {
	var lbfgs LBFGS
	lbfgs.Init(10, 2)
	var x, g Vector
	x.Init(2)
	g.Init(2)
	x.Set(0, 1)
	x.Set(1, 0.1)

	k := 0
	for {
		fmt.Println("======> k = ", k)
		fmt.Println("x = ", x)
		g.Set(0, 2*x.Get(0))
		g.Set(1, 2*x.Get(1))
		fmt.Println("g = ", g)
		delta := lbfgs.GetDeltaX(x, g)
		x = VecWeightedSum(x, delta, 1, 1)
		if g.Norm() < 0.0001 {
			break
		}
		k++
	}

	fmt.Println("==== done ====")
	fmt.Println("x = ", x)
}
