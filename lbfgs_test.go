package lbfgs

import (
	"fmt"
	"testing"
)

func TestOptimizer(t *testing.T) {
	optimizer := NewOptimizer(10, 2)

	x := NewVector(2)
	g := NewVector(2)
	x.SetValues([]float32{1, 0.1})

	k := 0
	for {
		fmt.Println("======> k = ", k)
		fmt.Println("x = ", x)
		g.SetValues([]float32{2 * x.Get(0), 2 * x.Get(1)})
		fmt.Println("g = ", g)
		delta := optimizer.GetDeltaX(x, g)
		x.Increment(delta, 1)
		if g.Norm() < 0.0001 {
			break
		}
		k++
	}

	fmt.Println("==== done ====")
	fmt.Println("x = ", x)
}
