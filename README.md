lbfgs
=====

L-BFGS的go语言实现

# 使用

下面的例子计算x^2+y^2在(1, 0.1)附近的极值。

```go
package main

import (
	"fmt"
	"github.com/huichen/lbfgs"
)

func main() {
	// 初始化LBFGS，存储步长为10，x向量为二维
	optimizer := lbfgs.NewOptimizer(10, 2)

	// x为自变量，g为目标函数的偏微分向量，都是二维向量
	x := lbfgs.NewVector(2)
	g := lbfgs.NewVector(2)

	// x初始化为(1, 0.1)
	x.SetValues([]float32{1, 0.1})

	k := 0
	for {
		fmt.Println("====== 第", k, "次迭代")
		fmt.Println("x =", x)

		// 更新偏导数向量
		g.SetValues([]float32{2*x.Get(0), 2*x.Get(1)})
		fmt.Println("g =", g)

		// 计算x更新的步长
		delta := optimizer.GetDeltaX(x, g)

		// 更新x
		x.Increment(delta, 1)

		// 检查是否满足收敛条件
		if g.Norm() < 0.0001 {
			break
		}
		k++
	}

	// 打印最终的x值
	fmt.Println("==== 结束 ====")
	fmt.Println("x =", x)
}
```
