#include "lfm.h"

template<class T, class K>
bool has_key(T &t, K k) {
    return (t.find(k) != t.end());
}

float lfm_random() {
    return float(random()) / float(RAND_MAX);
}

LFM::LFM(map<USERID, map<DOCID, LABELID> > &rating_data, int F, float alpha, float lmbd, int max_iter) {
    _F = F;
    _alpha = alpha;
    _lmbd = lmbd;
    _max_iter = max_iter;
    _rating_data = rating_data;

    map<DOCID, LABELID> *rates;
    map<USERID, map<DOCID, LABELID> >::iterator it;
    map<DOCID, LABELID>::iterator iter;
    for (it = rating_data.begin(); it != rating_data.end(); ++it) {
        rates = &(it->second);
        for (int i = 0; i < _F; ++i) {
            float rnd = lfm_random() / sqrt(_F);
            P[it->first].push_back(rnd);
        }
        for (iter = rates->begin(); iter != rates->end(); ++iter) {
            if (!has_key(Q, iter->first)) {
                for (int i = 0; i < _F; ++i) {
                    float rnd = lfm_random() / sqrt(_F);
                    Q[iter->first].push_back(rnd);
                }
            }
        }
    }
}

int LFM::train() {
    map<USERID, map<DOCID, LABELID> >::iterator it;
    map<DOCID, LABELID>::iterator iter;
    map<DOCID, LABELID> *rates;
    for (int i = 0; i < _max_iter; ++i) {
        for (it = _rating_data.begin(); it != _rating_data.end(); ++it) {
            rates = &(it->second);
            for (iter = rates->begin(); iter != rates->end(); ++iter) {
                float hat_rui = predict(it->first, iter->first);
                float err_ui = iter->second - hat_rui;
                lfm_info("%ld\t%d\t%f", it->first, iter->first, err_ui);
                for (int k = 0; k < _F; ++k) {
                    P[it->first][k] += _alpha * (err_ui * Q[iter->first][k] - _lmbd * P[it->first][k]);
                    Q[iter->first][k] += _alpha * (err_ui * P[it->first][k] - _lmbd * Q[iter->first][k]);
                }
            }
        }
        _alpha *= 0.9;
    }

    return 0;
}

float LFM::predict(USERID user, DOCID doc) {
    float score = 0.0;
    if (has_key(P, user) && has_key(Q, doc)) {
        for (int i = 0; i < _F; ++i)
            score += P[user][i] * Q[doc][i];
    }

    return score;
}
