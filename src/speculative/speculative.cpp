#include "speculative.h"

std::vector<int> find_candidate_draft(const std::vector<int>& tokens, int n, int k) {
    int S = tokens.size();
    if(S <= n) return {};
    for(int i = S - n - 1; i >= 0; i--) {
        bool match = true;
        for(int j = 0; j < n; j++) {
            if(tokens[i + j] != tokens[S - n + j]) {
                match = false;
                break;
            }
        }

        if(match) {
            std::vector<int> draft;
            for(int e = 0; e < k; e++) {
                int target_idx = i + n + e;
                if(target_idx >= S) break;
                draft.push_back(tokens[target_idx]);
            }
            return draft;
        }
    }

    return {};
}