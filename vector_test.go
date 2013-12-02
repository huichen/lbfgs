package lbfgs

import (
	"fmt"
	"testing"
)

func Expect(t *testing.T, expect string, actual interface{}) {
	actualString := fmt.Sprint(actual)
	if expect != actualString {
		t.Errorf("期待值=\"%s\", 实际=\"%s\"", expect, actualString)
	}
}

func TestVectorSetAndGet(t *testing.T) {
	vec := NewVector(10)
	Expect(t, "0", vec.Get(3))

	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))

	vec.Set(13, 1122)
	Expect(t, "1122", vec.Get(3))
	Expect(t, "1122", vec.Get(23))
}

func TestVectorOpposite(t *testing.T) {
	vec := NewVector(10)
	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))

	vec1 := vec.Opposite()
	Expect(t, "-7", vec1.Get(3))
}

func TestVectorNorm(t *testing.T) {
	vec := NewVector(2)
	vec.SetValues([]float32{3, 4})
	Expect(t, "5", vec.Norm())
}

func TestMod(t *testing.T) {
	Expect(t, "3", Mod(3, 4))
	Expect(t, "0", Mod(0, 4))
	Expect(t, "0", Mod(4, 4))
	Expect(t, "5", Mod(14, 9))
}

func TestVecDotProduct(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float32{1, 2, 3})
	vec2 := NewVector(3)
	vec2.SetValues([]float32{3, 4, 5})

	// 点乘积为 1*3+2*4+3*5 = 26
	Expect(t, "26", VecDotProduct(vec1, vec2))
}

func TestWeightedSum(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float32{1, 2, 3})
	vec2 := NewVector(3)
	vec2.SetValues([]float32{3, 4, 5})

	vec1.WeightedSum(vec1, vec2, 3, 4)
	Expect(t, "15", vec1.Get(0))
	Expect(t, "22", vec1.Get(1))
	Expect(t, "29", vec1.Get(2))
}

func TestDeepCopy(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float32{1, 2, 3})

	// shallow copy
	vec2 := vec1
	Expect(t, "1", vec2.Get(0))
	vec1.Set(0, 3)
	Expect(t, "3", vec2.Get(0))

	// deep copy
	vec3 := NewVector(3)
	vec3.DeepCopy(vec1)
	Expect(t, "3", vec3.Get(0))
	vec1.Set(0, 4)
	Expect(t, "3", vec3.Get(0))
}

func TestIncrement(t *testing.T) {
	vec1 := NewVector(3)
	vec1.SetValues([]float32{1, 2, 3})
	vec2 := NewVector(3)
	vec2.SetValues([]float32{1, 7, 2})

	vec1.Increment(vec2, 2)
	Expect(t, "3", vec1.Get(0))
	Expect(t, "16", vec1.Get(1))
	Expect(t, "7", vec1.Get(2))
}
