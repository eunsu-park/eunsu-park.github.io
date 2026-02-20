"""
수학과 정수론 (Number Theory)
Number Theory Algorithms

정수론의 기본 알고리즘을 구현합니다.
"""

from typing import List, Tuple
from functools import reduce
import math


# =============================================================================
# 1. GCD / LCM
# =============================================================================

def gcd(a: int, b: int) -> int:
    """
    최대공약수 (유클리드 호제법)
    시간복잡도: O(log(min(a, b)))
    """
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """최소공배수"""
    return a * b // gcd(a, b)


def gcd_multiple(numbers: List[int]) -> int:
    """여러 수의 GCD"""
    return reduce(gcd, numbers)


def lcm_multiple(numbers: List[int]) -> int:
    """여러 수의 LCM"""
    return reduce(lcm, numbers)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    확장 유클리드 알고리즘
    gcd(a, b) = a*x + b*y를 만족하는 (gcd, x, y) 반환
    """
    if b == 0:
        return a, 1, 0

    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1

    return g, x, y


# =============================================================================
# 2. 소수 (Prime Numbers)
# =============================================================================

def is_prime(n: int) -> bool:
    """
    소수 판별
    시간복잡도: O(√n)
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def sieve_of_eratosthenes(n: int) -> List[int]:
    """
    에라토스테네스의 체
    n 이하의 모든 소수 반환
    시간복잡도: O(n log log n)
    """
    if n < 2:
        return []

    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False

    for i in range(2, int(n ** 0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i * i, n + 1, i):
                is_prime_arr[j] = False

    return [i for i in range(n + 1) if is_prime_arr[i]]


def prime_factorization(n: int) -> List[Tuple[int, int]]:
    """
    소인수분해
    반환: [(소수, 지수), ...]
    시간복잡도: O(√n)
    """
    factors = []
    d = 2

    while d * d <= n:
        if n % d == 0:
            count = 0
            while n % d == 0:
                n //= d
                count += 1
            factors.append((d, count))
        d += 1

    if n > 1:
        factors.append((n, 1))

    return factors


# =============================================================================
# 3. 모듈러 연산 (Modular Arithmetic)
# =============================================================================

def mod_pow(base: int, exp: int, mod: int) -> int:
    """
    빠른 거듭제곱 (모듈러)
    시간복잡도: O(log exp)
    """
    result = 1
    base %= mod

    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp //= 2
        base = (base * base) % mod

    return result


def mod_inverse(a: int, mod: int) -> int:
    """
    모듈러 역원 (mod가 소수일 때)
    a^(-1) mod p = a^(p-2) mod p (페르마 소정리)
    """
    return mod_pow(a, mod - 2, mod)


def mod_inverse_extended(a: int, mod: int) -> int:
    """
    모듈러 역원 (확장 유클리드)
    gcd(a, mod) = 1일 때만 존재
    """
    g, x, _ = extended_gcd(a, mod)
    if g != 1:
        return -1  # 역원 없음
    return x % mod


# =============================================================================
# 4. 조합론 (Combinatorics)
# =============================================================================

def factorial_mod(n: int, mod: int) -> int:
    """n! mod p"""
    result = 1
    for i in range(2, n + 1):
        result = (result * i) % mod
    return result


class Combination:
    """
    조합 계산 (모듈러)
    전처리: O(n)
    쿼리: O(1)
    """

    def __init__(self, n: int, mod: int):
        self.mod = mod
        self.fact = [1] * (n + 1)
        self.inv_fact = [1] * (n + 1)

        # 팩토리얼 계산
        for i in range(1, n + 1):
            self.fact[i] = self.fact[i - 1] * i % mod

        # 역 팩토리얼 계산
        self.inv_fact[n] = mod_pow(self.fact[n], mod - 2, mod)
        for i in range(n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % mod

    def nCr(self, n: int, r: int) -> int:
        """nCr mod p"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod

    def nPr(self, n: int, r: int) -> int:
        """nPr mod p"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[n - r] % self.mod


# =============================================================================
# 5. 오일러 피 함수 (Euler's Totient)
# =============================================================================

def euler_phi(n: int) -> int:
    """
    φ(n) = n보다 작거나 같고 n과 서로소인 양의 정수의 개수
    시간복잡도: O(√n)
    """
    result = n

    d = 2
    while d * d <= n:
        if n % d == 0:
            while n % d == 0:
                n //= d
            result -= result // d
        d += 1

    if n > 1:
        result -= result // n

    return result


def euler_phi_sieve(n: int) -> List[int]:
    """1~n까지의 오일러 피 함수 값 (체)"""
    phi = list(range(n + 1))

    for i in range(2, n + 1):
        if phi[i] == i:  # i가 소수
            for j in range(i, n + 1, i):
                phi[j] -= phi[j] // i

    return phi


# =============================================================================
# 6. 중국인의 나머지 정리 (CRT)
# =============================================================================

def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """
    x ≡ r_i (mod m_i)를 만족하는 최소 양의 정수 x
    모든 m_i는 서로소여야 함
    """
    M = 1
    for m in moduli:
        M *= m

    result = 0
    for r, m in zip(remainders, moduli):
        Mi = M // m
        yi = mod_inverse_extended(Mi, m)
        result += r * Mi * yi

    return result % M


# =============================================================================
# 7. 이항 계수 (Lucas Theorem)
# =============================================================================

def lucas(n: int, r: int, p: int) -> int:
    """
    루카스 정리로 nCr mod p 계산
    p가 소수일 때, n, r이 매우 클 때 유용
    """
    if r == 0:
        return 1
    return lucas(n // p, r // p, p) * nCr_small(n % p, r % p, p) % p


def nCr_small(n: int, r: int, p: int) -> int:
    """작은 nCr mod p"""
    if r > n:
        return 0
    if r == 0 or r == n:
        return 1

    num = den = 1
    for i in range(r):
        num = num * (n - i) % p
        den = den * (i + 1) % p

    return num * mod_pow(den, p - 2, p) % p


# =============================================================================
# 8. 약수 (Divisors)
# =============================================================================

def divisors(n: int) -> List[int]:
    """
    n의 모든 약수
    시간복잡도: O(√n)
    """
    result = []

    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)

    return sorted(result)


def divisor_count(n: int) -> int:
    """약수의 개수"""
    factors = prime_factorization(n)
    count = 1
    for _, exp in factors:
        count *= (exp + 1)
    return count


def divisor_sum(n: int) -> int:
    """약수의 합"""
    factors = prime_factorization(n)
    total = 1
    for p, e in factors:
        total *= (pow(p, e + 1) - 1) // (p - 1)
    return total


# =============================================================================
# 9. 선형 디오판토스 방정식
# =============================================================================

def solve_linear_diophantine(a: int, b: int, c: int) -> Tuple[bool, int, int]:
    """
    ax + by = c의 정수해 (x, y) 찾기
    반환: (해 존재 여부, x, y)
    """
    g, x0, y0 = extended_gcd(abs(a), abs(b))

    if c % g != 0:
        return False, 0, 0

    x0 *= c // g
    y0 *= c // g

    if a < 0:
        x0 = -x0
    if b < 0:
        y0 = -y0

    return True, x0, y0


# =============================================================================
# 테스트
# =============================================================================

def main():
    print("=" * 60)
    print("수학과 정수론 (Number Theory) 예제")
    print("=" * 60)

    # 1. GCD / LCM
    print("\n[1] GCD / LCM")
    a, b = 48, 18
    print(f"    gcd({a}, {b}) = {gcd(a, b)}")
    print(f"    lcm({a}, {b}) = {lcm(a, b)}")
    g, x, y = extended_gcd(a, b)
    print(f"    확장 유클리드: {a}*{x} + {b}*{y} = {g}")

    # 2. 소수
    print("\n[2] 소수")
    print(f"    17 is prime: {is_prime(17)}")
    print(f"    18 is prime: {is_prime(18)}")
    primes = sieve_of_eratosthenes(50)
    print(f"    50 이하 소수: {primes}")
    print(f"    60의 소인수분해: {prime_factorization(60)}")

    # 3. 모듈러 연산
    print("\n[3] 모듈러 연산")
    mod = 1000000007
    print(f"    2^10 mod {mod} = {mod_pow(2, 10, mod)}")
    print(f"    3의 역원 mod 7 = {mod_inverse(3, 7)}")
    print(f"    검증: 3 * {mod_inverse(3, 7)} mod 7 = {3 * mod_inverse(3, 7) % 7}")

    # 4. 조합
    print("\n[4] 조합론")
    comb = Combination(1000, mod)
    print(f"    10C3 = {comb.nCr(10, 3)}")
    print(f"    10P3 = {comb.nPr(10, 3)}")
    print(f"    100C50 mod {mod} = {comb.nCr(100, 50)}")

    # 5. 오일러 피 함수
    print("\n[5] 오일러 피 함수")
    for n in [10, 12, 7, 1]:
        print(f"    φ({n}) = {euler_phi(n)}")

    # 6. 중국인의 나머지 정리
    print("\n[6] 중국인의 나머지 정리")
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    x = chinese_remainder_theorem(remainders, moduli)
    print(f"    x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)")
    print(f"    x = {x}")
    print(f"    검증: {x % 3}, {x % 5}, {x % 7}")

    # 7. 루카스 정리
    print("\n[7] 루카스 정리")
    n, r, p = 1000, 500, 13
    print(f"    C({n}, {r}) mod {p} = {lucas(n, r, p)}")

    # 8. 약수
    print("\n[8] 약수")
    n = 36
    print(f"    {n}의 약수: {divisors(n)}")
    print(f"    약수 개수: {divisor_count(n)}")
    print(f"    약수 합: {divisor_sum(n)}")

    # 9. 디오판토스 방정식
    print("\n[9] 선형 디오판토스 방정식")
    a, b, c = 3, 5, 7
    exists, x, y = solve_linear_diophantine(a, b, c)
    print(f"    {a}x + {b}y = {c}")
    print(f"    해 존재: {exists}, x={x}, y={y}")
    print(f"    검증: {a}*{x} + {b}*{y} = {a * x + b * y}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
