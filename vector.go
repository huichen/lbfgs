package lbfgs

import (
	"log"
	"math"
)

type vector struct {
	// 向量长度
	length int

	// 向量中元素的值
	values []float32
}

// 构造长度（维度）为length的向量
// vector为私有类型，因此只能通过这个函数进行创建
func NewVector(length int) *vector {
	v := new(vector)
	v.length = length
	v.values = make([]float32, length)
	v.Clear()
	return v
}

func (v *vector) Clear() {
	for i := 0; i < v.length; i++ {
		v.values[i] = 0.0
	}
}

// 设置向量中所有元素的值
func (v *vector) SetValues(values []float32) {
	if v.length != len(values) {
		log.Fatal("SetValues参数切片长度和向量长度不一致")
	}
	for i := 0; i < v.length; i++ {
		v.values[i] = values[i]
	}
}

// 设置向量中元素的值
func (v *vector) Set(index int, value float32) {
	v.values[Mod(index, v.length)] = value
}

// 得到向量中元素的值
func (v *vector) Get(index int) float32 {
	return v.values[Mod(index, v.length)]
}

// 得到 -vector
func (v *vector) Opposite() *vector {
	output := NewVector(v.length)
	for i := 0; i < v.length; i++ {
		output.values[i] = -v.values[i]
	}
	return output
}

// 复制元素的值
func (v *vector) DeepCopy(that *vector) {
	if v.length != that.length {
		log.Fatal("赋值时向量长度不匹配")
	}
	for i := 0; i < v.length; i++ {
		v.values[i] = that.values[i]
	}
}

// v = v + alpha * that
func (v *vector) Increment(that *vector, alpha float32) {
	if v.length != that.length {
		log.Fatal("Increment时向量长度不匹配")
	}
	for i := 0; i < v.length; i++ {
		v.values[i] += that.values[i] * alpha
	}
}

// 1-模
func (v *vector) Norm() float32 {
	var result float32
	for i := 0; i < v.length; i++ {
		result += v.values[i] * v.values[i]
	}
	return float32(math.Sqrt(float64(result)))
}

// 计算两个向量的线性求和 v = a * vector1 + b * vector2
func (v *vector) WeightedSum(vector1, vector2 *vector, a, b float32) {
	if v.length != vector1.length || vector1.length != vector2.length {
		log.Fatal("v, vector1和vector2的长度不一致")
	}

	for iterVec1 := 0; iterVec1 < vector1.length; iterVec1++ {
		v.Set(iterVec1, a*vector1.values[iterVec1]+b*vector2.values[iterVec1])
	}
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
func VecDotProduct(vector1, vector2 *vector) float32 {
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
