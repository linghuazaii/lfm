#include "lfm.h"
#include <fstream>
#include <unistd.h>
#include <sstream>
#include <algorithm>

template<class T, class V>
void strip(T &t, V v) {
    std::remove(t.begin(), t.end(), v);
}

void split(string &str, char delim, vector<string> &result) {
    result.clear();
    stringstream ss(str);
    string token;
    while (getline(ss, token, delim)) {
        if (token == "")
            continue;
        result.push_back(token);
    }
}

void prepare_data(const char *train, map<USERID, map<DOCID, LABELID> > &result) {
    ifstream file;
    file.open(train, ios::in);
    string line;
    vector<string> elem;
    while (getline(file, line)) {
        strip(line, ' ');
        split(line, '\t', elem);
        if (elem.size() != 3) {
            continue;
        }
        USERID user = atoll(elem[0].c_str());
        DOCID doc = atoi(elem[1].c_str());
        LABELID label = atoi(elem[2].c_str());
        result[user][doc] = label;
    }
    file.close();
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: lfm <train data file> <test data file> <pred result file>\n");
        exit(1);
    }
    map<USERID, map<DOCID, LABELID> > rating_data;
    prepare_data(argv[1], rating_data);
    LFM lfm(rating_data, 32);
    lfm.train();

    map<USERID, map<DOCID, LABELID> > test_data;
    prepare_data(argv[2], test_data);

    FILE *pred_file = fopen(argv[3], "w+");

    map<USERID, map<DOCID, LABELID> >::iterator it;
    map<DOCID, LABELID>::iterator iter;
    for (it = test_data.begin(); it != test_data.end(); ++it) {
        for (iter = it->second.begin(); iter != it->second.end(); ++iter) {
            float pred = lfm.predict(it->first, iter->first);
            fprintf(pred_file, "%ld\t%d\t%f\n", it->first, iter->first, pred);
        }
    }

    fclose(pred_file);

    return 0;
}
