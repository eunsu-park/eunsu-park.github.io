/*
 * 배열과 문자열 (Array and String)
 * Two Pointers, Sliding Window, Prefix Sum
 *
 * 배열과 문자열 처리의 핵심 기법들입니다.
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>

using namespace std;

// =============================================================================
// 1. 투 포인터 (Two Pointers)
// =============================================================================

// 정렬된 배열에서 두 수의 합
pair<int, int> twoSumSorted(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;

    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target)
            return {left, right};
        else if (sum < target)
            left++;
        else
            right--;
    }

    return {-1, -1};
}

// 물 담기 (Container With Most Water)
int maxArea(const vector<int>& height) {
    int left = 0, right = height.size() - 1;
    int maxWater = 0;

    while (left < right) {
        int h = min(height[left], height[right]);
        int w = right - left;
        maxWater = max(maxWater, h * w);

        if (height[left] < height[right])
            left++;
        else
            right--;
    }

    return maxWater;
}

// 세 수의 합
vector<vector<int>> threeSum(vector<int>& nums) {
    vector<vector<int>> result;
    sort(nums.begin(), nums.end());
    int n = nums.size();

    for (int i = 0; i < n - 2; i++) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;

        int left = i + 1, right = n - 1;
        while (left < right) {
            int sum = nums[i] + nums[left] + nums[right];
            if (sum == 0) {
                result.push_back({nums[i], nums[left], nums[right]});
                while (left < right && nums[left] == nums[left + 1]) left++;
                while (left < right && nums[right] == nums[right - 1]) right--;
                left++;
                right--;
            } else if (sum < 0) {
                left++;
            } else {
                right--;
            }
        }
    }

    return result;
}

// =============================================================================
// 2. 슬라이딩 윈도우 (Sliding Window)
// =============================================================================

// 고정 크기 윈도우 최대 합
int maxSumWindow(const vector<int>& arr, int k) {
    int n = arr.size();
    if (n < k) return -1;

    int windowSum = 0;
    for (int i = 0; i < k; i++) {
        windowSum += arr[i];
    }

    int maxSum = windowSum;
    for (int i = k; i < n; i++) {
        windowSum += arr[i] - arr[i - k];
        maxSum = max(maxSum, windowSum);
    }

    return maxSum;
}

// 중복 없는 최장 부분 문자열
int lengthOfLongestSubstring(const string& s) {
    unordered_set<char> charSet;
    int maxLen = 0;
    int left = 0;

    for (int right = 0; right < (int)s.length(); right++) {
        while (charSet.count(s[right])) {
            charSet.erase(s[left]);
            left++;
        }
        charSet.insert(s[right]);
        maxLen = max(maxLen, right - left + 1);
    }

    return maxLen;
}

// 합이 target 이상인 최소 길이 부분 배열
int minSubArrayLen(int target, const vector<int>& nums) {
    int n = nums.size();
    int minLen = INT_MAX;
    int sum = 0;
    int left = 0;

    for (int right = 0; right < n; right++) {
        sum += nums[right];
        while (sum >= target) {
            minLen = min(minLen, right - left + 1);
            sum -= nums[left];
            left++;
        }
    }

    return minLen == INT_MAX ? 0 : minLen;
}

// =============================================================================
// 3. 프리픽스 합 (Prefix Sum)
// =============================================================================

class PrefixSum {
private:
    vector<long long> prefix;

public:
    PrefixSum(const vector<int>& arr) {
        int n = arr.size();
        prefix.resize(n + 1, 0);
        for (int i = 0; i < n; i++) {
            prefix[i + 1] = prefix[i] + arr[i];
        }
    }

    // [left, right] 구간 합
    long long rangeSum(int left, int right) {
        return prefix[right + 1] - prefix[left];
    }
};

// 합이 k인 부분 배열 개수
int subarraySum(const vector<int>& nums, int k) {
    unordered_map<int, int> prefixCount;
    prefixCount[0] = 1;
    int sum = 0;
    int count = 0;

    for (int num : nums) {
        sum += num;
        if (prefixCount.count(sum - k)) {
            count += prefixCount[sum - k];
        }
        prefixCount[sum]++;
    }

    return count;
}

// =============================================================================
// 4. 카데인 알고리즘 (Kadane's Algorithm)
// =============================================================================

int maxSubArray(const vector<int>& nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];

    for (size_t i = 1; i < nums.size(); i++) {
        currentSum = max(nums[i], currentSum + nums[i]);
        maxSum = max(maxSum, currentSum);
    }

    return maxSum;
}

// 최대 부분 배열의 인덱스도 반환
tuple<int, int, int> maxSubArrayWithIndices(const vector<int>& nums) {
    int maxSum = nums[0];
    int currentSum = nums[0];
    int start = 0, end = 0, tempStart = 0;

    for (size_t i = 1; i < nums.size(); i++) {
        if (nums[i] > currentSum + nums[i]) {
            currentSum = nums[i];
            tempStart = i;
        } else {
            currentSum += nums[i];
        }

        if (currentSum > maxSum) {
            maxSum = currentSum;
            start = tempStart;
            end = i;
        }
    }

    return {maxSum, start, end};
}

// =============================================================================
// 5. 문자열 처리
// =============================================================================

// 팰린드롬 체크
bool isPalindrome(const string& s) {
    int left = 0, right = s.length() - 1;
    while (left < right) {
        if (s[left] != s[right]) return false;
        left++;
        right--;
    }
    return true;
}

// 애너그램 체크
bool isAnagram(const string& s, const string& t) {
    if (s.length() != t.length()) return false;

    vector<int> count(26, 0);
    for (size_t i = 0; i < s.length(); i++) {
        count[s[i] - 'a']++;
        count[t[i] - 'a']--;
    }

    for (int c : count) {
        if (c != 0) return false;
    }
    return true;
}

// 문자열 압축
string compress(const string& s) {
    if (s.empty()) return s;

    string result;
    int count = 1;

    for (size_t i = 1; i <= s.length(); i++) {
        if (i < s.length() && s[i] == s[i - 1]) {
            count++;
        } else {
            result += s[i - 1];
            if (count > 1) {
                result += to_string(count);
            }
            count = 1;
        }
    }

    return result.length() < s.length() ? result : s;
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "배열과 문자열 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 투 포인터
    cout << "\n[1] 투 포인터" << endl;
    vector<int> sorted = {1, 2, 3, 4, 6};
    auto [idx1, idx2] = twoSumSorted(sorted, 6);
    cout << "    배열: [1,2,3,4,6], 타겟: 6" << endl;
    cout << "    인덱스: (" << idx1 << ", " << idx2 << ")" << endl;

    vector<int> heights = {1, 8, 6, 2, 5, 4, 8, 3, 7};
    cout << "    물 담기: " << maxArea(heights) << endl;

    // 2. 슬라이딩 윈도우
    cout << "\n[2] 슬라이딩 윈도우" << endl;
    vector<int> arr = {1, 4, 2, 10, 23, 3, 1, 0, 20};
    cout << "    크기 4 윈도우 최대 합: " << maxSumWindow(arr, 4) << endl;

    cout << "    \"abcabcbb\" 최장 부분 문자열: " << lengthOfLongestSubstring("abcabcbb") << endl;

    vector<int> nums = {2, 3, 1, 2, 4, 3};
    cout << "    합 >= 7 최소 길이: " << minSubArrayLen(7, nums) << endl;

    // 3. 프리픽스 합
    cout << "\n[3] 프리픽스 합" << endl;
    vector<int> prefixArr = {1, 2, 3, 4, 5};
    PrefixSum ps(prefixArr);
    cout << "    배열: [1,2,3,4,5]" << endl;
    cout << "    [1,3] 구간 합: " << ps.rangeSum(1, 3) << endl;

    vector<int> subarr = {1, 1, 1};
    cout << "    합이 2인 부분 배열 개수: " << subarraySum(subarr, 2) << endl;

    // 4. 카데인 알고리즘
    cout << "\n[4] 카데인 알고리즘" << endl;
    vector<int> kadane = {-2, 1, -3, 4, -1, 2, 1, -5, 4};
    cout << "    배열: [-2,1,-3,4,-1,2,1,-5,4]" << endl;
    cout << "    최대 부분 배열 합: " << maxSubArray(kadane) << endl;

    // 5. 문자열 처리
    cout << "\n[5] 문자열 처리" << endl;
    cout << "    \"racecar\" 팰린드롬: " << (isPalindrome("racecar") ? "예" : "아니오") << endl;
    cout << "    \"anagram\"/\"nagaram\" 애너그램: " << (isAnagram("anagram", "nagaram") ? "예" : "아니오") << endl;
    cout << "    \"aabcccccaaa\" 압축: " << compress("aabcccccaaa") << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 알고리즘        | 시간복잡도 |" << endl;
    cout << "    |-----------------|------------|" << endl;
    cout << "    | 투 포인터       | O(n)       |" << endl;
    cout << "    | 슬라이딩 윈도우 | O(n)       |" << endl;
    cout << "    | 프리픽스 합     | O(1) 쿼리  |" << endl;
    cout << "    | 카데인          | O(n)       |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
