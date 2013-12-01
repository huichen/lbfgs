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

func TestVectorInit(t *testing.T) {
	var vec Vector
	vec.Init(10)
}

func TestVectorSetAndGet(t *testing.T) {
	var vec Vector
	vec.Init(10)
	Expect(t, "0", vec.Get(3))

	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))

	vec.Set(13, 1122)
	Expect(t, "1122", vec.Get(3))
	Expect(t, "1122", vec.Get(23))
}

func TestVectorOpposite(t *testing.T) {
	var vec Vector
	vec.Init(10)
	vec.Set(3, 7)
	Expect(t, "7", vec.Get(3))

	vec1 := vec.Opposite()
	Expect(t, "-7", vec1.Get(3))
}

func TestVectorNorm(t *testing.T) {
	var vec Vector
	vec.Init(2)
	vec.Set(0, 3)
	vec.Set(1, 4)
	Expect(t, "5", vec.Norm())
}

func TestMod(t *testing.T) {
	Expect(t, "3", Mod(3, 4))
	Expect(t, "0", Mod(0, 4))
	Expect(t, "0", Mod(4, 4))
	Expect(t, "5", Mod(14, 9))
}

func TestVecDotProduct(t *testing.T) {
	var vec1, vec2 Vector
	vec1.Init(3)
	vec2.Init(3)

	vec1.Set(0, 1)
	vec1.Set(1, 2)
	vec1.Set(2, 3)

	vec2.Set(0, 3)
	vec2.Set(1, 4)
	vec2.Set(2, 5)

	// 点乘积为 1*3+2*4+3*5 = 26
	Expect(t, "26", VecDotProduct(vec1, vec2))
}

func TestVecWeightedSum(t *testing.T) {
	var vec1, vec2 Vector
	vec1.Init(3)
	vec2.Init(3)

	vec1.Set(0, 1)
	vec1.Set(1, 2)
	vec1.Set(2, 3)

	vec2.Set(0, 3)
	vec2.Set(1, 4)
	vec2.Set(2, 5)

	vec := VecWeightedSum(vec1, vec2, 3, 4)
	Expect(t, "15", vec.Get(0))
	Expect(t, "22", vec.Get(1))
	Expect(t, "29", vec.Get(2))
}
