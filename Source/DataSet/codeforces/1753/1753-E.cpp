// bai!

#include <bits/stdc++.h>

using i64 = long long;

constexpr i64 inf = 4E18;

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    int n, b, p, m;
    std::cin >> n >> b >> p >> m;
    
    std::vector<int> x;
    int total = 0;
    
    std::vector<std::vector<int>> a;
    std::vector<int> mul;
    for (int i = 0; i < n; i++) {
        char t;
        int y;
        std::cin >> t >> y;
        
        if (t == '+') {
            x.push_back(y);
            total++;
        } else if (y > 1) {
            a.push_back(std::move(x));
            mul.push_back(y);
        }
    }
    
    a.push_back(std::move(x));
    
    int cntm = mul.size();
    
    for (int i = 0; i <= cntm; i++) {
        std::sort(a[i].begin(), a[i].end());
    }
    
    std::vector<std::vector<i64>> suf(cntm + 1);
    for (int i = 0; i <= cntm; i++) {
        suf[i].resize(a[i].size() + 1);
        for (int j = int(a[i].size()) - 1; j >= 0; j--) {
            suf[i][j] = suf[i][j + 1] + a[i][j];
        }
    }
    
    i64 ans = 0;
    
    std::map<int, std::vector<int>> mpos;
    for (int i = 0; i < cntm; i++) {
        mpos[mul[i]].push_back(i);
    }
    
    std::vector<bool> move(cntm);
    auto dfs = [&](auto self, auto it) {
        if (it == mpos.end()) {
            std::vector<i64> val(cntm + 1, 1);
            i64 cost = 0;
            for (int i = 0; i < cntm; i++) {
                if (move[i]) {
                    val[cntm] *= mul[i];
                    cost += m;
                } else {
                    val[i] *= mul[i];
                }
            }
            for (int i = cntm - 1; i >= 0; i--) {
                val[i] *= val[i + 1];
            }
            if (cost > b) {
                return;
            }
            const int rest = std::min((b - cost) / p, 1LL * total);
            i64 cur = val[0];
            for (int i = 0; i <= cntm; i++) {
                cur += val[i] * suf[i][0];
            }
            
            auto get = [&](auto x) {
                int cnt = 0;
                i64 sum = 0;
                for (int i = 0; i <= cntm; i++) {
                    if (val[i] == val[0]) {
                        continue;
                    }
                    i64 y = x / (val[0] - val[i]);
                    auto it = std::upper_bound(a[i].begin(), a[i].end(), y);
                    int k = it - a[i].begin();
                    cnt += a[i].size() - k;
                    sum += suf[i][k] * (val[0] - val[i]);
                }
                return std::pair(cnt, sum);
            };
            
            i64 lo = 0, hi = inf;
            while (lo < hi) {
                i64 x = (lo + hi) / 2;
                if (get(x).first > rest) {
                    lo = x + 1;
                } else {
                    hi = x;
                }
            }
            auto [c, s] = get(lo);
            cur += s + 1LL * (rest - c) * lo;
            ans = std::max(ans, cur);
            return;
        }
        self(self, std::next(it));
        for (int i = 0; i < int(it->second.size()); i++) {
            move[it->second[i]] = true;
            self(self, std::next(it));
        }
        for (auto i : it->second) {
            move[i] = false;
        }
    };
    dfs(dfs, mpos.begin());
    std::cout << ans << "\n";
    
    return 0;
}
