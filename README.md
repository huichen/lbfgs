lbfgs
=====

L-BFGS的go语言实现

# 使用

下面的例子计算x^2+y^2在(1, 0.1)附近的极值。

```go
import (
	"github.com/huichen/lbfgs"
)

func main() {
	// 初始化LBFGS，存储步长为10，x向量为二维
        var optimizer lbfgs.LBFGS
        optimizer.Init(10, 2)

	// x为自变量，g为目标函数的偏微分向量
        var x, g lbfgs.Vector

	// x和g需要初始化，见Vector.Init函数
        x.Init(2)
        g.Init(2)

	// 给x赋初值(1, 0.1)
        x.Set(0, 1)
        x.Set(1, 0.1)

        k := 0
        for {
                fmt.Println("======> k = ", k)
                fmt.Println("x = ", x)

		// 更新偏导数向量
                g.Set(0, 2*x.Get(0))
                g.Set(1, 2*x.Get(1))
                fmt.Println("g = ", g)

		// 计算x更新的步长
                delta := optimizer.GetDeltaX(x, g)

		// 更新x
                x = lbfgs.VecWeightedSum(x, delta, 1, 1)

		// 检查是否满足收敛条件
                if g.Norm() < 0.0001 {
                        break
                }
                k++
        }

	// 打印最终的x值
        fmt.Println("==== 结束 ====")
        fmt.Println("x = ", x)
}
```
