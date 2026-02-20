/*
 * 정수론 (Number Theory)
 * GCD/LCM, Prime, Modular Arithmetic, Combinatorics
 *
 * 수학적 문제 해결을 위한 알고리즘입니다.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

const long long MOD = 1e9 + 7;

// =============================================================================
// 1. GCD / LCM
// =============================================================================

long long gcd(long long a, long long b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}

long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b;
}

// 확장 유클리드 알고리즘
// ax + by = gcd(a, b)를 만족하는 x, y 반환
tuple<long long, long long, long long> extendedGcd(long long a, long long b) {
    if (b == 0) {
        return {a, 1, 0};
    }
    auto [g, x1, y1] = extendedGcd(b, a % b);
    return {g, y1, x1 - (a / b) * y1};
}

// =============================================================================
// 2. 소수 판정 및 생성
// =============================================================================

bool isPrime(long long n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    for (long long i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// 에라토스테네스의 체
vector<bool> sieveOfEratosthenes(int n) {
    vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;

    for (int i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }

    return isPrime;
}

// 소인수분해
vector<pair<long long, int>> factorize(long long n) {
    vector<pair<long long, int>> factors;

    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            int cnt = 0;
            while (n % i == 0) {
                n /= i;
                cnt++;
            }
            factors.push_back({i, cnt});
        }
    }

    if (n > 1) {
        factors.push_back({n, 1});
    }

    return factors;
}

// =============================================================================
// 3. 모듈러 연산
// =============================================================================

// (a + b) % mod
long long addMod(long long a, long long b, long long mod = MOD) {
    return ((a % mod) + (b % mod)) % mod;
}

// (a * b) % mod
long long mulMod(long long a, long long b, long long mod = MOD) {
    return ((a % mod) * (b % mod)) % mod;
}

// (a ^ b) % mod (빠른 거듭제곱)
long long powMod(long long a, long long b, long long mod = MOD) {
    long long result = 1;
    a %= mod;

    while (b > 0) {
        if (b & 1) {
            result = mulMod(result, a, mod);
        }
        a = mulMod(a, a, mod);
        b >>= 1;
    }

    return result;
}

// 모듈러 역원 (페르마의 소정리, mod가 소수일 때)
long long modInverse(long long a, long long mod = MOD) {
    return powMod(a, mod - 2, mod);
}

// 모듈러 역원 (확장 유클리드)
long long modInverseExtGcd(long long a, long long mod) {
    auto [g, x, y] = extendedGcd(a, mod);
    if (g != 1) return -1;  // 역원 없음
    return (x % mod + mod) % mod;
}

// =============================================================================
// 4. 조합론
// =============================================================================

class Combination {
private:
    vector<long long> fact;
    vector<long long> invFact;
    long long mod;

public:
    Combination(int n, long long mod = MOD) : mod(mod) {
        fact.resize(n + 1);
        invFact.resize(n + 1);

        fact[0] = 1;
        for (int i = 1; i <= n; i++) {
            fact[i] = mulMod(fact[i-1], i, mod);
        }

        invFact[n] = powMod(fact[n], mod - 2, mod);
        for (int i = n - 1; i >= 0; i--) {
            invFact[i] = mulMod(invFact[i+1], i + 1, mod);
        }
    }

    // nCr
    long long C(int n, int r) {
        if (r < 0 || r > n) return 0;
        return mulMod(fact[n], mulMod(invFact[r], invFact[n-r], mod), mod);
    }

    // nPr
    long long P(int n, int r) {
        if (r < 0 || r > n) return 0;
        return mulMod(fact[n], invFact[n-r], mod);
    }

    // 카탈란 수
    long long catalan(int n) {
        return mulMod(C(2 * n, n), modInverse(n + 1, mod), mod);
    }
};

// =============================================================================
// 5. 오일러 피 함수
// =============================================================================

long long eulerPhi(long long n) {
    long long result = n;

    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0) {
                n /= i;
            }
            result -= result / i;
        }
    }

    if (n > 1) {
        result -= result / n;
    }

    return result;
}

// 1~n까지 오일러 피 함수
vector<int> eulerPhiSieve(int n) {
    vector<int> phi(n + 1);
    iota(phi.begin(), phi.end(), 0);

    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {  // i가 소수
            for (int j = i; j <= n; j += i) {
                phi[j] -= phi[j] / i;
            }
        }
    }

    return phi;
}

// =============================================================================
// 6. 중국인의 나머지 정리 (CRT)
// =============================================================================

// x ≡ a1 (mod m1), x ≡ a2 (mod m2)
pair<long long, long long> crt(long long a1, long long m1,
                                long long a2, long long m2) {
    auto [g, p, q] = extendedGcd(m1, m2);

    if ((a2 - a1) % g != 0) {
        return {-1, -1};  // 해 없음
    }

    long long l = m1 / g * m2;  // lcm
    long long x = ((a1 + m1 * (((a2 - a1) / g * p) % (m2 / g))) % l + l) % l;

    return {x, l};
}

// =============================================================================
// 7. 밀러-라빈 소수 판정 (큰 수)
// =============================================================================

long long mulModLarge(long long a, long long b, long long mod) {
    return (__int128)a * b % mod;
}

long long powModLarge(long long a, long long b, long long mod) {
    long long result = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) result = mulModLarge(result, a, mod);
        a = mulModLarge(a, a, mod);
        b >>= 1;
    }
    return result;
}

bool millerRabin(long long n, long long a) {
    if (n % a == 0) return n == a;

    long long d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }

    long long x = powModLarge(a, d, n);
    if (x == 1 || x == n - 1) return true;

    for (int i = 0; i < r - 1; i++) {
        x = mulModLarge(x, x, n);
        if (x == n - 1) return true;
    }

    return false;
}

bool isPrimeLarge(long long n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    vector<long long> witnesses = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (long long a : witnesses) {
        if (n == a) return true;
        if (!millerRabin(n, a)) return false;
    }
    return true;
}

// =============================================================================
// 8. 약수 관련
// =============================================================================

// 약수 목록
vector<long long> getDivisors(long long n) {
    vector<long long> divisors;

    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.push_back(i);
            if (i != n / i) {
                divisors.push_back(n / i);
            }
        }
    }

    sort(divisors.begin(), divisors.end());
    return divisors;
}

// 약수 개수
int countDivisors(long long n) {
    int count = 0;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            count++;
            if (i != n / i) count++;
        }
    }
    return count;
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "정수론 예제" << endl;
    cout << "============================================================" << endl;

    // 1. GCD / LCM
    cout << "\n[1] GCD / LCM" << endl;
    cout << "    gcd(48, 18) = " << gcd(48, 18) << endl;
    cout << "    lcm(48, 18) = " << lcm(48, 18) << endl;
    auto [g, x, y] = extendedGcd(48, 18);
    cout << "    48*" << x << " + 18*" << y << " = " << g << endl;

    // 2. 소수
    cout << "\n[2] 소수 판정" << endl;
    cout << "    17 is prime: " << (isPrime(17) ? "yes" : "no") << endl;
    cout << "    100 is prime: " << (isPrime(100) ? "yes" : "no") << endl;

    auto sieve = sieveOfEratosthenes(30);
    cout << "    30 이하 소수: ";
    for (int i = 2; i <= 30; i++) {
        if (sieve[i]) cout << i << " ";
    }
    cout << endl;

    // 3. 소인수분해
    cout << "\n[3] 소인수분해" << endl;
    cout << "    360 = ";
    auto factors = factorize(360);
    for (size_t i = 0; i < factors.size(); i++) {
        cout << factors[i].first << "^" << factors[i].second;
        if (i < factors.size() - 1) cout << " × ";
    }
    cout << endl;

    // 4. 모듈러 연산
    cout << "\n[4] 모듈러 연산" << endl;
    cout << "    2^10 mod 1000 = " << powMod(2, 10, 1000) << endl;
    cout << "    3의 역원 mod 7 = " << modInverse(3, 7) << endl;

    // 5. 조합론
    cout << "\n[5] 조합론" << endl;
    Combination comb(100);
    cout << "    C(10, 3) = " << comb.C(10, 3) << endl;
    cout << "    P(10, 3) = " << comb.P(10, 3) << endl;
    cout << "    Catalan(5) = " << comb.catalan(5) << endl;

    // 6. 오일러 피 함수
    cout << "\n[6] 오일러 피 함수" << endl;
    cout << "    φ(12) = " << eulerPhi(12) << endl;
    cout << "    φ(36) = " << eulerPhi(36) << endl;

    // 7. CRT
    cout << "\n[7] 중국인의 나머지 정리" << endl;
    auto [result, mod] = crt(2, 3, 3, 5);
    cout << "    x ≡ 2 (mod 3), x ≡ 3 (mod 5)" << endl;
    cout << "    x = " << result << " (mod " << mod << ")" << endl;

    // 8. 약수
    cout << "\n[8] 약수" << endl;
    cout << "    36의 약수: ";
    for (auto d : getDivisors(36)) {
        cout << d << " ";
    }
    cout << endl;
    cout << "    36의 약수 개수: " << countDivisors(36) << endl;

    // 9. 복잡도 요약
    cout << "\n[9] 복잡도 요약" << endl;
    cout << "    | 알고리즘          | 시간복잡도        |" << endl;
    cout << "    |-------------------|-------------------|" << endl;
    cout << "    | GCD (유클리드)    | O(log min(a,b))   |" << endl;
    cout << "    | 소수 판정         | O(√n)             |" << endl;
    cout << "    | 에라토스테네스    | O(n log log n)    |" << endl;
    cout << "    | 소인수분해        | O(√n)             |" << endl;
    cout << "    | 빠른 거듭제곱     | O(log n)          |" << endl;
    cout << "    | 밀러-라빈         | O(k log² n)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
