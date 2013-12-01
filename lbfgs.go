package lbfgs

// limited-memory BFGS 实现
//
// l-bfgs的迭代算法见下面的论文
//   Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited Storage".
//   Mathematics of Computation 35 (151): 773–782. doi:10.1090/S0025-5718-1980-0572855-7
//
// 这种方法最多保存最近m步的中间结果用以计算海森矩阵的近似值。
type LBFGS struct {
	// 保存的最大步长数
	m int

	// 向量维度，也就是特征的个数
	dimension int

	// 当前的步数，从0开始
	// 如果需要重新优化，请调用Clear函数
	k int

	// 自变量
	x []Vector

	// 目标函数的偏导数向量
	g []Vector

	// s_k = x_(k+1) - x_k
	s []Vector

	// y_k = g_(k+1) - g_k
	y []Vector

	// ro_k = 1 / Y_k .* s_k
	ro Vector
}

// 初始化LBFGS计算结构体
// steps为保存的步长数, dimension为x的维度（特征数）
func (l *LBFGS) Init(steps int, dimension int) {
	l.m = steps
	l.dimension = dimension
	l.x = make([]Vector, l.m)
	l.g = make([]Vector, l.m)
	l.s = make([]Vector, l.m)
	l.y = make([]Vector, l.m)
	l.ro.Init(l.m)
	l.k = 0

	for i := 0; i < l.m; i++ {
		l.x[i].Init(dimension)
		l.g[i].Init(dimension)
		l.s[i].Init(dimension)
		l.y[i].Init(dimension)
	}
}

// 清除结构体中保存的数据，以便重复使用结构体
func (l *LBFGS) Clear() {
	for i := 0; i < l.m; i++ {
		l.x[i].Clear()
		l.g[i].Clear()
		l.s[i].Clear()
		l.y[i].Clear()
	}
	l.ro.Clear()
	l.k = 0
}

// 输入x_k和g_k，返回x需要更新的增量 d_k = - H_k * g_k
func (l *LBFGS) GetDeltaX(x, g Vector) Vector {
	currIndex := Mod(l.k, l.m)

	// 更新x_k
	l.x[currIndex].Assign(x)

	// 更新g_k
	l.g[currIndex].Assign(g)

	// 当为第0步时，使用简单的带学习率的gradient descent
	if l.k == 0 {
		var vec Vector
		vec.Init(l.dimension)
		for i := 0; i < l.dimension; i++ {
			vec.Set(i, -g.Get(i)*0.0001)
		}
		l.k++
		return vec
	}

	prevIndex := Mod(l.k-1, l.m)

	// 更新s_(k-1)
	l.s[prevIndex] = VecWeightedSum(l.x[currIndex], l.x[prevIndex], 1, -1)

	// 更新y_(k-1)
	l.y[prevIndex] = VecWeightedSum(l.g[currIndex], l.g[prevIndex], 1, -1)

	// 更新ro_(k-1)
	l.ro.Set(l.k-1, 1.0/VecDotProduct(l.y[prevIndex], l.s[prevIndex]))

	// 计算两重循环的下限
	lowerBound := l.k - l.m
	if lowerBound < 0 {
		lowerBound = 0
	}

	// 第一重循环
	q := g
	var alpha Vector
	alpha.Init(l.m)
	for i := l.k - 1; i >= lowerBound; i-- {
		currIndex := Mod(i, l.m)
		alpha.Set(i, l.ro.Get(i)*VecDotProduct(l.s[currIndex], q))
		q = VecWeightedSum(q, l.y[currIndex], 1, -alpha.Get(i))
	}

	// 第二重循环
	z := q
	var beta Vector
	beta.Init(l.m)
	for i := lowerBound; i <= l.k-1; i++ {
		currIndex := Mod(i, l.m)
		beta.Set(i, l.ro.Get(i)*VecDotProduct(l.y[currIndex], z))
		z = VecWeightedSum(z, l.s[currIndex], 1, alpha.Get(i)-beta.Get(i))
	}

	// 更新k
	l.k++

	return z.Opposite()
}
