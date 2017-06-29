#ifndef _LATENT_FACTOR_MODEL_H
#define _LATENT_FACTOR_MODEL_H
/**
 * File: lfm.h
 * Date: 2017-6-29
 */
#include <stdio.h>
#include <inttypes.h>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>
using namespace std;

#define lfm_info(fmt, ...) do {\
    printf(fmt, ##__VA_ARGS__);\
    printf("\n");\
} while (0)

typedef uint64_t USERID;
typedef uint32_t DOCID;
typedef uint32_t LABELID;

class LFM {
public:
    LFM(map<USERID, map<DOCID, LABELID> > &rating_data, int F, float alpha = 0.1, float lmbd = 0.1, int max_iter = 50);
    int train();
    float predict(USERID user, DOCID doc);
private:
    map<USERID, map<DOCID, LABELID> > _rating_data;
    map<USERID, vector<float> > P;
    map<DOCID, vector<float> > Q;
    int _F;
    float _alpha;
    float _lmbd;
    int _max_iter;
};

#endif
