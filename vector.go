package lbfgs

import (
	"log"
	"math"
)

type Vector struct {
	// 向量长度
	length int

	// 向量中元素的值
	values []float32

	// 向量是否已经初始化
	initialized bool
}

// 向量在使用前必须初始化，向量长度一经确定无法更改
func (vector *Vector) Init(length int) {
	if vector.initialized {
		log.Fatal("Vector不能初始化两次")
	}
	vector.length = length
	vector.values = make([]float32, length)
	for i := 0; i < length; i++ {
		vector.values[i] = 0.0
	}
	vector.initialized = true
}

func (vector *Vector) Clear() {
	vector.CheckInit()
	for i := 0; i < vector.length; i++ {
		vector.values[i] = 0.0
	}
}

// 检查向量是否已经初始化
func (vector *Vector) CheckInit() {
	if !vector.initialized {
		log.Fatal("vector没有初始化")
	}
}

// 设置向量中元素的值
func (vector *Vector) Set(index int, value float32) {
	vector.CheckInit()
	vector.values[Mod(index, vector.length)] = value
}

// 得到向量中元素的值
func (vector *Vector) Get(index int) float32 {
	vector.CheckInit()
	return vector.values[Mod(index, vector.length)]
}

// 得到 -vector
func (vector *Vector) Opposite() Vector {
	vector.CheckInit()
	var output Vector
	output.Init(vector.length)
	for i := 0; i < vector.length; i++ {
		output.values[i] = -vector.values[i]
	}
	return output
}

// 赋值
func (vector *Vector) Assign(that Vector) {
	vector.CheckInit()
	if vector.length != that.length {
		log.Fatal("赋值时向量长度不匹配")
	}
	for i := 0; i < vector.length; i++ {
		vector.values[i] = that.values[i]
	}
}

// 1-模
func (vector *Vector) Norm() float32 {
	vector.CheckInit()
	var result float32
	for i := 0; i < vector.length; i++ {
		result += vector.values[i] * vector.values[i]
	}
	return float32(math.Sqrt(float64(result)))
}

///////////////////////////////////////////////////////////////////////////////
/*
  工具函数
*/
///////////////////////////////////////////////////////////////////////////////

// 计算 a mod b
func Mod(a, b int) int {
	if b <= 0 || a < 0 {
		log.Fatal("模运算的值不在合法范围")
	}
	return a - a/b*b
}

// 计算点乘积 vector1^T * vector2
func VecDotProduct(vector1, vector2 Vector) float32 {
	vector1.CheckInit()
	vector2.CheckInit()
	if vector1.length != vector2.length {
		log.Fatal("vector1和vector2的长度不一致")
	}

	var result float32
	result = 0
	for iterVec1 := 0; iterVec1 < vector1.length; iterVec1++ {
		result += vector1.values[iterVec1] * vector2.values[iterVec1]
	}
	return result
}

// 计算点乘积 a * vector1 + b * vector2
func VecWeightedSum(vector1, vector2 Vector, a, b float32) Vector {
	vector1.CheckInit()
	vector2.CheckInit()
	if vector1.length != vector2.length {
		log.Fatal("vector1和vector2的长度不一致")
	}

	var vec Vector
	vec.Init(vector1.length)
	for iterVec1 := 0; iterVec1 < vector1.length; iterVec1++ {
		vec.Set(iterVec1, a*vector1.values[iterVec1]+b*vector2.values[iterVec1])
	}
	return vec
}
