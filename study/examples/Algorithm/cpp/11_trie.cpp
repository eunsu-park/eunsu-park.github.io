/*
 * 트라이 (Trie)
 * Prefix Tree, Autocomplete, Word Search
 *
 * 문자열 검색에 특화된 트리 자료구조입니다.
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

using namespace std;

// =============================================================================
// 1. 기본 트라이
// =============================================================================

class Trie {
private:
    struct TrieNode {
        unordered_map<char, unique_ptr<TrieNode>> children;
        bool isEndOfWord = false;
    };

    unique_ptr<TrieNode> root;

public:
    Trie() : root(make_unique<TrieNode>()) {}

    void insert(const string& word) {
        TrieNode* node = root.get();
        for (char c : word) {
            if (!node->children.count(c)) {
                node->children[c] = make_unique<TrieNode>();
            }
            node = node->children[c].get();
        }
        node->isEndOfWord = true;
    }

    bool search(const string& word) const {
        const TrieNode* node = root.get();
        for (char c : word) {
            if (!node->children.count(c)) {
                return false;
            }
            node = node->children.at(c).get();
        }
        return node->isEndOfWord;
    }

    bool startsWith(const string& prefix) const {
        const TrieNode* node = root.get();
        for (char c : prefix) {
            if (!node->children.count(c)) {
                return false;
            }
            node = node->children.at(c).get();
        }
        return true;
    }
};

// =============================================================================
// 2. 자동완성 트라이
// =============================================================================

class AutocompleteTrie {
private:
    struct TrieNode {
        unordered_map<char, unique_ptr<TrieNode>> children;
        bool isEndOfWord = false;
        int frequency = 0;
    };

    unique_ptr<TrieNode> root;

    void collectWords(TrieNode* node, string& current, vector<string>& result) {
        if (node->isEndOfWord) {
            result.push_back(current);
        }
        for (auto& [c, child] : node->children) {
            current.push_back(c);
            collectWords(child.get(), current, result);
            current.pop_back();
        }
    }

public:
    AutocompleteTrie() : root(make_unique<TrieNode>()) {}

    void insert(const string& word, int freq = 1) {
        TrieNode* node = root.get();
        for (char c : word) {
            if (!node->children.count(c)) {
                node->children[c] = make_unique<TrieNode>();
            }
            node = node->children[c].get();
        }
        node->isEndOfWord = true;
        node->frequency += freq;
    }

    vector<string> autocomplete(const string& prefix) {
        vector<string> result;
        TrieNode* node = root.get();

        for (char c : prefix) {
            if (!node->children.count(c)) {
                return result;
            }
            node = node->children[c].get();
        }

        string current = prefix;
        collectWords(node, current, result);
        return result;
    }
};

// =============================================================================
// 3. 와일드카드 검색 트라이
// =============================================================================

class WildcardTrie {
private:
    struct TrieNode {
        TrieNode* children[26] = {nullptr};
        bool isEndOfWord = false;
        ~TrieNode() {
            for (auto& child : children) delete child;
        }
    };

    TrieNode* root;

    bool searchHelper(TrieNode* node, const string& word, int idx) {
        if (!node) return false;
        if (idx == (int)word.size()) return node->isEndOfWord;

        char c = word[idx];
        if (c == '.') {
            // 와일드카드: 모든 자식 탐색
            for (int i = 0; i < 26; i++) {
                if (searchHelper(node->children[i], word, idx + 1)) {
                    return true;
                }
            }
            return false;
        } else {
            return searchHelper(node->children[c - 'a'], word, idx + 1);
        }
    }

public:
    WildcardTrie() : root(new TrieNode()) {}
    ~WildcardTrie() { delete root; }

    void addWord(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            int idx = c - 'a';
            if (!node->children[idx]) {
                node->children[idx] = new TrieNode();
            }
            node = node->children[idx];
        }
        node->isEndOfWord = true;
    }

    bool search(const string& word) {
        return searchHelper(root, word, 0);
    }
};

// =============================================================================
// 4. 단어 사전 (Word Break)
// =============================================================================

class WordDictionary {
private:
    struct TrieNode {
        unordered_map<char, TrieNode*> children;
        bool isEnd = false;
    };

    TrieNode* root;

public:
    WordDictionary() : root(new TrieNode()) {}

    void addWord(const string& word) {
        TrieNode* node = root;
        for (char c : word) {
            if (!node->children.count(c)) {
                node->children[c] = new TrieNode();
            }
            node = node->children[c];
        }
        node->isEnd = true;
    }

    bool wordBreak(const string& s) {
        int n = s.size();
        vector<bool> dp(n + 1, false);
        dp[0] = true;

        for (int i = 1; i <= n; i++) {
            TrieNode* node = root;
            for (int j = i - 1; j >= 0 && node; j--) {
                node = node->children.count(s[j]) ? node->children[s[j]] : nullptr;
                if (node && node->isEnd && dp[j]) {
                    dp[i] = true;
                    break;
                }
            }
        }

        return dp[n];
    }
};

// =============================================================================
// 5. XOR 트라이 (최대 XOR 찾기)
// =============================================================================

class XORTrie {
private:
    struct TrieNode {
        TrieNode* children[2] = {nullptr, nullptr};
    };

    TrieNode* root;
    static const int MAX_BITS = 31;

public:
    XORTrie() : root(new TrieNode()) {}

    void insert(int num) {
        TrieNode* node = root;
        for (int i = MAX_BITS; i >= 0; i--) {
            int bit = (num >> i) & 1;
            if (!node->children[bit]) {
                node->children[bit] = new TrieNode();
            }
            node = node->children[bit];
        }
    }

    int getMaxXOR(int num) {
        TrieNode* node = root;
        int maxXor = 0;
        for (int i = MAX_BITS; i >= 0; i--) {
            int bit = (num >> i) & 1;
            int oppositeBit = 1 - bit;

            if (node->children[oppositeBit]) {
                maxXor |= (1 << i);
                node = node->children[oppositeBit];
            } else if (node->children[bit]) {
                node = node->children[bit];
            } else {
                break;
            }
        }
        return maxXor;
    }
};

int findMaximumXOR(const vector<int>& nums) {
    XORTrie trie;
    int maxXor = 0;

    for (int num : nums) {
        trie.insert(num);
        maxXor = max(maxXor, trie.getMaxXOR(num));
    }

    return maxXor;
}

// =============================================================================
// 테스트
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "트라이 예제" << endl;
    cout << "============================================================" << endl;

    // 1. 기본 트라이
    cout << "\n[1] 기본 트라이" << endl;
    Trie trie;
    trie.insert("apple");
    trie.insert("app");
    trie.insert("application");
    cout << "    삽입: apple, app, application" << endl;
    cout << "    search(\"apple\"): " << (trie.search("apple") ? "있음" : "없음") << endl;
    cout << "    search(\"app\"): " << (trie.search("app") ? "있음" : "없음") << endl;
    cout << "    search(\"appl\"): " << (trie.search("appl") ? "있음" : "없음") << endl;
    cout << "    startsWith(\"app\"): " << (trie.startsWith("app") ? "예" : "아니오") << endl;

    // 2. 자동완성
    cout << "\n[2] 자동완성" << endl;
    AutocompleteTrie autoTrie;
    autoTrie.insert("hello");
    autoTrie.insert("help");
    autoTrie.insert("helicopter");
    autoTrie.insert("hero");
    autoTrie.insert("world");

    cout << "    단어: hello, help, helicopter, hero, world" << endl;
    auto suggestions = autoTrie.autocomplete("hel");
    cout << "    \"hel\" 자동완성: ";
    for (const auto& s : suggestions) cout << s << " ";
    cout << endl;

    // 3. 와일드카드 검색
    cout << "\n[3] 와일드카드 검색" << endl;
    WildcardTrie wcTrie;
    wcTrie.addWord("bad");
    wcTrie.addWord("dad");
    wcTrie.addWord("mad");
    cout << "    단어: bad, dad, mad" << endl;
    cout << "    search(\".ad\"): " << (wcTrie.search(".ad") ? "있음" : "없음") << endl;
    cout << "    search(\"b..\"): " << (wcTrie.search("b..") ? "있음" : "없음") << endl;

    // 4. 단어 분리
    cout << "\n[4] 단어 분리 (Word Break)" << endl;
    WordDictionary dict;
    dict.addWord("leet");
    dict.addWord("code");
    cout << "    사전: [leet, code]" << endl;
    cout << "    \"leetcode\" 분리 가능: " << (dict.wordBreak("leetcode") ? "예" : "아니오") << endl;

    // 5. 최대 XOR
    cout << "\n[5] 최대 XOR" << endl;
    vector<int> nums = {3, 10, 5, 25, 2, 8};
    cout << "    배열: [3, 10, 5, 25, 2, 8]" << endl;
    cout << "    최대 XOR: " << findMaximumXOR(nums) << endl;

    // 6. 복잡도 요약
    cout << "\n[6] 복잡도 요약" << endl;
    cout << "    | 연산        | 시간복잡도 | L: 문자열 길이  |" << endl;
    cout << "    |-------------|------------|-----------------|" << endl;
    cout << "    | 삽입        | O(L)       |                 |" << endl;
    cout << "    | 검색        | O(L)       |                 |" << endl;
    cout << "    | 접두사 검색 | O(L)       |                 |" << endl;
    cout << "    | 자동완성    | O(L + K)   | K: 결과 수      |" << endl;
    cout << "    | 공간복잡도  | O(ALPHABET × L × N) |      |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
