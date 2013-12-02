package lbfgs

import (
	"log"
)

// limited-memory BFGS 实现
//
// l-bfgs的迭代算法见下面的论文
//   Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage".
//   Mathematics of Computation 35 (151): 773–782. doi:10.1090/S0025-5718-1980-0572855-7
//
// 这种方法最多保存最近m步的中间结果用以计算海森矩阵的近似值。
//
// 请用NewOptimizer函数建立新的优化器。
//
// 注意optimizer是
//   1. 协程不安全的，请为每个协程建一个optimizer；
//   2. 迭代不安全的，optimizer会记录每个迭代步骤的中间结果，如果需要重新开始新的优化，
//      请调用Clear函数。
type optimizer struct {
	// 保存的最大步长数
	m int

	// 向量维度，也就是特征的个数
	dimension int

	// 当前的步数，从0开始
	// 如果需要重新优化，请调用Clear函数
	k int

	// 自变量
	x []*vector

	// 目标函数的偏导数向量
	g []*vector

	// s_k = x_(k+1) - x_k
	s []*vector

	// y_k = g_(k+1) - g_k
	y []*vector

	// ro_k = 1 / Y_k .* s_k
	ro *vector

	// 临时变量
	q, z, alpha, beta *vector
}

// 初始化优化结构体
// steps为保存的步长数, dimension为x的维度（特征数）
func NewOptimizer(steps int, dimension int) *optimizer {
	opt := new(optimizer)
	opt.m = steps
	opt.dimension = dimension
	opt.k = 0

	opt.x = make([]*vector, opt.m)
	opt.g = make([]*vector, opt.m)
	opt.s = make([]*vector, opt.m)
	opt.y = make([]*vector, opt.m)
	opt.ro = NewVector(opt.m)
	opt.q = NewVector(opt.dimension)
	opt.z = NewVector(opt.dimension)
	opt.alpha = NewVector(opt.m)
	opt.beta = NewVector(opt.m)

	for i := 0; i < opt.m; i++ {
		opt.x[i] = NewVector(dimension)
		opt.g[i] = NewVector(dimension)
		opt.s[i] = NewVector(dimension)
		opt.y[i] = NewVector(dimension)
	}
	return opt
}

// 清除结构体中保存的数据，以便重复使用结构体
func (opt *optimizer) Clear() {
	for i := 0; i < opt.m; i++ {
		opt.x[i].Clear()
		opt.g[i].Clear()
		opt.s[i].Clear()
		opt.y[i].Clear()
	}
	opt.ro.Clear()
	opt.q.Clear()
	opt.z.Clear()
	opt.alpha.Clear()
	opt.beta.Clear()
	opt.k = 0
}

// 输入x_k和g_k，返回x需要更新的增量 d_k = - H_k * g_k
func (opt *optimizer) GetDeltaX(x, g *vector) *vector {
	if x.length != opt.dimension || g.length != opt.dimension {
		log.Fatal("x或者g的维度和optimzier维度不一致")
	}

	currIndex := Mod(opt.k, opt.m)

	// 更新x_k
	opt.x[currIndex].DeepCopy(x)

	// 更新g_k
	opt.g[currIndex].DeepCopy(g)

	// 当为第0步时，使用简单的带学习率的gradient descent
	if opt.k == 0 {
		vec := NewVector(opt.dimension)
		for i := 0; i < opt.dimension; i++ {
			vec.Set(i, -g.Get(i)*0.0001)
		}
		opt.k++
		return vec
	}

	prevIndex := Mod(opt.k-1, opt.m)

	// 更新s_(k-1)
	opt.s[prevIndex].WeightedSum(opt.x[currIndex], opt.x[prevIndex], 1, -1)

	// 更新y_(k-1)
	opt.y[prevIndex].WeightedSum(opt.g[currIndex], opt.g[prevIndex], 1, -1)

	// 更新ro_(k-1)
	opt.ro.Set(opt.k-1, 1.0/VecDotProduct(opt.y[prevIndex], opt.s[prevIndex]))

	// 计算两个循环的下限
	lowerBound := opt.k - opt.m
	if lowerBound < 0 {
		lowerBound = 0
	}

	// 第一个循环
	opt.q.DeepCopy(g)
	for i := opt.k - 1; i >= lowerBound; i-- {
		currIndex := Mod(i, opt.m)
		opt.alpha.Set(i, opt.ro.Get(i)*VecDotProduct(opt.s[currIndex], opt.q))
		opt.q.Increment(opt.y[currIndex], -opt.alpha.Get(i))
	}

	// 第二个循环
	opt.z.DeepCopy(opt.q)
	for i := lowerBound; i <= opt.k-1; i++ {
		currIndex := Mod(i, opt.m)
		opt.beta.Set(i, opt.ro.Get(i)*VecDotProduct(opt.y[currIndex], opt.z))
		opt.z.Increment(opt.s[currIndex], opt.alpha.Get(i)-opt.beta.Get(i))
	}

	// 更新k
	opt.k++

	return opt.z.Opposite()
}
